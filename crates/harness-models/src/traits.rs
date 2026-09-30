use harness_attention::KVCache;
use harness_core::{ModelConfig, Result, Tensor};

/// Unified interface for frontier LLMs
pub trait CausalLM: Send + Sync {
    /// Model architectural configuration
    fn config(&self) -> &ModelConfig;

    /// Embed token IDs into hidden state vectors [seq_len, hidden_dim]
    fn embed(&self, token_ids: &[u32]) -> Result<Tensor>;

    /// Forward pass through an individual layer (allows layer-by-layer offload/streaming)
    fn forward_layer(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor>;

    /// Project final hidden states to vocabulary logits [seq_len, vocab_size]
    fn lm_head(&self, hidden_states: &Tensor) -> Result<Tensor>;

    /// Complete forward pass for prefill or decode token
    fn forward(
        &self,
        token_ids: &[u32],
        start_pos: usize,
        kv_caches: &mut [KVCache],
    ) -> Result<Tensor> {
        let mut h = self.embed(token_ids)?;
        let num_layers = self.config().num_hidden_layers;

        for (layer_idx, kv_cache) in kv_caches.iter_mut().enumerate().take(num_layers) {
            h = self.forward_layer(layer_idx, &h, start_pos, kv_cache)?;
        }

        self.lm_head(&h)
    }
}
