use harness_core::{MoEConfig, Result, Tensor};

/// Gating network that computes top-k router probabilities for active experts
pub struct MoERouter {
    gate_weight: Tensor, // [hidden_dim, num_routed_experts]
    top_k: usize,
    norm_topk_prob: bool,
}

impl MoERouter {
    pub fn new(gate_weight: Tensor, config: &MoEConfig) -> Self {
        Self {
            gate_weight,
            top_k: config.routing_top_k,
            norm_topk_prob: config.norm_topk_prob,
        }
    }

    /// Select top-k experts per token and return expert weights
    pub fn route(&self, hidden_states: &Tensor) -> Result<(Vec<Vec<usize>>, Vec<Vec<f32>>)> {
        // Compute router logits = hidden_states @ gate_weight
        let logits = hidden_states.matmul(&self.gate_weight)?;
        let logits_slice = logits.as_f32_slice()?;
        let num_tokens = hidden_states.shape()[0];
        let num_experts = *self.gate_weight.shape().last().unwrap_or(&1);

        let mut token_expert_indices = Vec::with_capacity(num_tokens);
        let mut token_expert_weights = Vec::with_capacity(num_tokens);

        for t in 0..num_tokens {
            let offset = t * num_experts;
            let token_logits = &logits_slice[offset..offset + num_experts];

            // Softmax over all experts
            let max_val = token_logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = token_logits.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let probs: Vec<f32> = exps.iter().map(|&e| e / sum_exp.max(1e-8)).collect();

            // Find top-k expert indices
            let mut indexed_probs: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
            indexed_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            let selected: Vec<(usize, f32)> = indexed_probs.into_iter().take(self.top_k).collect();
            let indices: Vec<usize> = selected.iter().map(|(idx, _)| *idx).collect();
            let mut weights: Vec<f32> = selected.iter().map(|(_, w)| *w).collect();

            if self.norm_topk_prob {
                let sum_w: f32 = weights.iter().sum();
                for w in &mut weights {
                    *w /= sum_w.max(1e-8);
                }
            }

            token_expert_indices.push(indices);
            token_expert_weights.push(weights);
        }

        Ok((token_expert_indices, token_expert_weights))
    }
}

pub struct MoELayer {
    pub router: MoERouter,
    pub num_experts: usize,
    pub hidden_dim: usize,
    pub intermediate_dim: usize,
}
