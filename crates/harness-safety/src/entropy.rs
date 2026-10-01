use harness_core::{Result, Tensor};

pub struct EntropyDetector {
    pub anomaly_threshold: f32, // Typically 2.5 nats
}

impl EntropyDetector {
    pub fn new(anomaly_threshold: f32) -> Self {
        Self { anomaly_threshold }
    }

    /// Compute Shannon entropy over logit distribution: H(p) = - sum(p * log(p))
    pub fn compute_entropy(logits: &Tensor) -> Result<f32> {
        let logits_slice = logits.as_f32_slice()?;
        if logits_slice.is_empty() {
            return Ok(0.0);
        }
        let max_logit = logits_slice.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        let mut sum_exp = 0.0f32;
        let exps: Vec<f32> = logits_slice
            .iter()
            .map(|&l| {
                let e = (l - max_logit).exp();
                sum_exp += e;
                e
            })
            .collect();

        let inv_sum = 1.0 / sum_exp.max(1e-8);
        let mut entropy = 0.0f32;

        for &e in &exps {
            let p = e * inv_sum;
            if p > 1e-8 {
                entropy -= p * p.ln();
            }
        }

        Ok(entropy)
    }

    pub fn is_hallucination_risk(&self, entropy: f32) -> bool {
        entropy > self.anomaly_threshold
    }
}
