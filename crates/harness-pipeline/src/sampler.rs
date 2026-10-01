use harness_core::{Result, Tensor};
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub min_p: f32,
    pub repetition_penalty: f32,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_p: 0.9,
            top_k: 40,
            min_p: 0.05,
            repetition_penalty: 1.1,
        }
    }
}

pub struct TokenSampler;

impl TokenSampler {
    /// Sample a token ID from raw logits given sampling hyperparameters and an optional boolean logit mask
    pub fn sample(
        logits: &Tensor,
        config: &SamplingConfig,
        mask: Option<&[bool]>,
        generated_history: &[u32],
    ) -> Result<u32> {
        let logits_slice = logits.as_f32_slice()?;
        let _vocab_size = logits_slice.len();

        let mut filtered: Vec<(usize, f32)> = logits_slice
            .iter()
            .copied()
            .enumerate()
            .collect();

        // 1. Apply repetition penalty
        if (config.repetition_penalty - 1.0).abs() > 1e-4 {
            let mut seen = std::collections::HashSet::new();
            for &tok in generated_history {
                if seen.insert(tok) {
                    if let Some((_, logit)) = filtered.get_mut(tok as usize) {
                        if *logit > 0.0 {
                            *logit /= config.repetition_penalty;
                        } else {
                            *logit *= config.repetition_penalty;
                        }
                    }
                }
            }
        }

        // 2. Apply constrained decoding logit validity mask
        if let Some(valid_mask) = mask {
            for (idx, (_, logit)) in filtered.iter_mut().enumerate() {
                if idx < valid_mask.len() && !valid_mask[idx] {
                    *logit = f32::NEG_INFINITY;
                }
            }
        }

        // 3. Greedy sampling if temperature is near zero
        if config.temperature <= 1e-4 {
            let best = filtered
                .iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| *idx as u32)
                .unwrap_or(0);
            return Ok(best);
        }

        // 4. Scale by temperature
        for (_, logit) in &mut filtered {
            *logit /= config.temperature;
        }

        // 5. Softmax
        let max_logit = filtered.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
        if max_logit == f32::NEG_INFINITY {
            return Err(harness_core::HarnessError::ConstrainedDecoding(
                "No valid tokens remain after applying constraints and masks".into(),
            ));
        }
        let mut sum_exp = 0.0f32;
        let mut probs: Vec<(usize, f32)> = filtered
            .iter()
            .map(|&(idx, l)| {
                let p = (l - max_logit).exp();
                sum_exp += p;
                (idx, p)
            })
            .collect();

        for (_, p) in &mut probs {
            *p /= sum_exp.max(1e-8);
        }

        // 6. Sort descending for top-p and top-k filtering
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // 7. Top-K cutoff
        if config.top_k > 0 && config.top_k < probs.len() {
            probs.truncate(config.top_k);
        }

        // 8. Top-P (nucleus) cutoff
        if config.top_p < 1.0 {
            let mut cum_p = 0.0f32;
            let mut cutoff_idx = probs.len();
            for (i, &(_, p)) in probs.iter().enumerate() {
                cum_p += p;
                if cum_p >= config.top_p {
                    cutoff_idx = i + 1;
                    break;
                }
            }
            probs.truncate(cutoff_idx);
        }

        // 9. Renormalize & random choice
        let total_p = probs.iter().map(|(_, p)| *p).sum::<f32>();
        if total_p <= 1e-8 {
            return Ok(probs.first().map(|(idx, _)| *idx as u32).unwrap_or(0));
        }

        let mut rng = rand::thread_rng();
        let r: f32 = rng.gen_range(0.0..total_p);
        let mut acc = 0.0f32;

        for &(idx, p) in &probs {
            acc += p;
            if acc >= r {
                return Ok(idx as u32);
            }
        }

        Ok(probs.last().map(|(idx, _)| *idx as u32).unwrap_or(0))
    }
}
