use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama4,
    Qwen3,
    DeepSeekV4,
    Phi4,
    Gemma3,
    Mistral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum QuantizationMode {
    None,
    FP8,
    INT8,
    Q4_K_M,
    Q3_K_M,
    NF4,
    ISQ, // In-Situ Dynamic Quantization
}

/// MoE (Mixture of Experts) Hyperparameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MoEConfig {
    pub num_routed_experts: usize,
    pub num_shared_experts: usize,
    pub num_active_experts: usize,
    pub routing_top_k: usize,
    pub norm_topk_prob: bool,
}

/// Universal architecture hyperparameters configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub moe: Option<MoEConfig>,
    pub quantization: QuantizationMode,
}

impl ModelConfig {
    /// Recommended configuration for Qwen 3.8-27B
    pub fn qwen3_27b() -> Self {
        Self {
            architecture: ModelArchitecture::Qwen3,
            vocab_size: 152064,
            hidden_size: 5120,
            intermediate_size: 27648,
            num_hidden_layers: 64,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            moe: None,
            quantization: QuantizationMode::Q4_K_M,
        }
    }

    /// Configuration for 70B Frontier Dense / MoE (e.g. Llama 4 Scout / 70B)
    pub fn llama4_70b() -> Self {
        Self {
            architecture: ModelArchitecture::Llama4,
            vocab_size: 128256,
            hidden_size: 8192,
            intermediate_size: 28672,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            moe: None,
            quantization: QuantizationMode::Q4_K_M,
        }
    }

    /// Configuration for DeepSeek V4 Sparse MoE
    pub fn deepseek_v4_flash() -> Self {
        Self {
            architecture: ModelArchitecture::DeepSeekV4,
            vocab_size: 129280,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 60,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            max_position_embeddings: 1048576,
            rms_norm_eps: 1e-6,
            rope_theta: 100000.0,
            moe: Some(MoEConfig {
                num_routed_experts: 64,
                num_shared_experts: 2,
                num_active_experts: 6,
                routing_top_k: 6,
                norm_topk_prob: true,
            }),
            quantization: QuantizationMode::FP8,
        }
    }
}
