use crate::moe::MoERouter;
use crate::traits::CausalLM;
use crate::transformer::TransformerBlock;
use harness_attention::KVCache;
use harness_core::{ModelConfig, Result, Tensor};

pub struct DeepSeekV4Model {
    pub config: ModelConfig,
    pub token_embedding: Tensor,
    pub dense_layers: Vec<TransformerBlock>,
    pub moe_routers: Vec<MoERouter>,
    pub final_norm: Tensor,
    pub lm_head_weight: Tensor,
}

impl CausalLM for DeepSeekV4Model {
    fn config(&self) -> &ModelConfig {
        &self.config
    }

    fn embed(&self, token_ids: &[u32]) -> Result<Tensor> {
        let hidden_dim = self.config.hidden_size;
        let mut out = vec![0.0f32; token_ids.len() * hidden_dim];
        let emb_slice = self.token_embedding.as_f32_slice()?;

        for (i, &tok) in token_ids.iter().enumerate() {
            let tok_idx = tok as usize;
            if tok_idx >= self.config.vocab_size {
                return Err(harness_core::HarnessError::InvalidShape(format!(
                    "Token ID {} exceeds model vocabulary size {}",
                    tok, self.config.vocab_size
                )));
            }
            let offset = tok_idx * hidden_dim;
            let src = &emb_slice[offset..offset + hidden_dim];
            let dst = &mut out[i * hidden_dim..(i + 1) * hidden_dim];
            dst.copy_from_slice(src);
        }

        Tensor::from_f32_slice(
            &out,
            vec![token_ids.len(), hidden_dim],
            self.token_embedding.device().clone(),
        )
    }

    fn forward_layer(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        self.dense_layers[layer_idx].forward(hidden_states, start_pos, kv_cache)
    }

    fn lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let normed = hidden_states.rms_norm(&self.final_norm, self.config.rms_norm_eps)?;
        normed.matmul(&self.lm_head_weight)
    }
}
