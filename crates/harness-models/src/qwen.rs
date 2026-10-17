use crate::traits::CausalLM;
use crate::transformer::TransformerBlock;
use harness_attention::KVCache;
use harness_core::{MoEConfig, ModelArchitecture, ModelConfig, QuantizationMode, Result, Tensor};

/// Unified Qwen2.5-Coder (32B / 14B / 7B / 1.5B) & Qwen3 (32B Dense / 30B-A3B MoE) Model
pub struct Qwen3Model {
    pub config: ModelConfig,
    pub token_embedding: Tensor,
    pub layers: Vec<TransformerBlock>,
    pub final_norm: Tensor,
    pub lm_head_weight: Tensor,
    pub tie_word_embeddings: bool,
}

pub type Qwen2_5CoderModel = Qwen3Model;

impl Qwen3Model {
    /// Canonical configuration for local `qwen2.5-coder:32b` GGUF (`Q4_K_M`, 64 layers, GQA 40:8)
    pub fn qwen2_5_coder_32b_config() -> ModelConfig {
        ModelConfig {
            architecture: ModelArchitecture::Qwen2_5,
            vocab_size: 152064,
            hidden_size: 5120,
            intermediate_size: 27648,
            num_hidden_layers: 64,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            moe: None,
            quantization: QuantizationMode::Q4_K_M,
        }
    }

    /// Canonical configuration for local `qwen2.5-coder:1.5b` Drafter GGUF (`Q4_K_M`, 28 layers, GQA 12:2, tied embeddings)
    pub fn qwen2_5_coder_1_5b_draft_config() -> ModelConfig {
        ModelConfig {
            architecture: ModelArchitecture::Qwen2_5,
            vocab_size: 151936,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: 2,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            moe: None,
            quantization: QuantizationMode::Q4_K_M,
        }
    }

    /// Canonical configuration for `Qwen3-30B-A3B-Instruct` Sparse MoE (30.5B total, 3.05B active/token, 128 experts, top-8)
    pub fn qwen3_30b_a3b_moe_config() -> ModelConfig {
        ModelConfig {
            architecture: ModelArchitecture::Qwen3,
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 768,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            moe: Some(MoEConfig {
                num_routed_experts: 128,
                num_shared_experts: 0,
                num_active_experts: 8,
                routing_top_k: 8,
                norm_topk_prob: true,
            }),
            quantization: QuantizationMode::Q4_K_M,
        }
    }
}

impl CausalLM for Qwen3Model {
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
        self.layers[layer_idx].forward(hidden_states, start_pos, kv_cache)
    }

    fn lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let normed = hidden_states.rms_norm(&self.final_norm, self.config.rms_norm_eps)?;
        if self.tie_word_embeddings {
            normed.matmul(&self.token_embedding)
        } else {
            normed.matmul(&self.lm_head_weight)
        }
    }
}

