use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Llama3,
    Llama4,
    Qwen2_5,
    Qwen3,
    DeepSeekV3,
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
    /// Primary 32B Coder Sweet-Spot Configuration: Qwen 2.5 Coder 32B Instruct (64 Layers, GQA 40:8, 32.76B Params)
    pub fn qwen2_5_coder_32b() -> Self {
        Self {
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

    /// 100% VRAM-Resident Speculative Drafter for Qwen 2.5 Coder 32B: Qwen 2.5 Coder 1.5B Instruct (28 Layers, GQA 12:2)
    pub fn qwen2_5_coder_1_5b_draft() -> Self {
        Self {
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

    /// Mid-Tier Dense Coder Configuration: Qwen 2.5 Coder 14B Instruct (48 Layers, GQA 40:8, 14.7B Params)
    pub fn qwen2_5_coder_14b() -> Self {
        Self {
            architecture: ModelArchitecture::Qwen2_5,
            vocab_size: 152064,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 48,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            max_position_embeddings: 32768,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            moe: None,
            quantization: QuantizationMode::Q4_K_M,
        }
    }

    /// High-Speed Sparse Coder MoE: Qwen3-30B-A3B (30.5B Total, 3.0B Active Parameters, 128 Routed Experts, Top-8)
    pub fn qwen3_30b_a3b_moe() -> Self {
        Self {
            architecture: ModelArchitecture::Qwen3,
            vocab_size: 151936,
            hidden_size: 2048,
            intermediate_size: 768,
            num_hidden_layers: 48,
            num_attention_heads: 32,
            num_key_value_heads: 4,
            max_position_embeddings: 32768,
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

    /// DeepSeek Coder 33B Instruct (62 Layers, GQA 56:8, 7168 Hidden Dim)
    pub fn deepseek_coder_33b() -> Self {
        Self {
            architecture: ModelArchitecture::DeepSeekV3,
            vocab_size: 32256,
            hidden_size: 7168,
            intermediate_size: 19200,
            num_hidden_layers: 62,
            num_attention_heads: 56,
            num_key_value_heads: 8,
            max_position_embeddings: 16384,
            rms_norm_eps: 1e-6,
            rope_theta: 100000.0,
            moe: None,
            quantization: QuantizationMode::Q4_K_M,
        }
    }

    /// Gemma 2 27B Instruct (46 Layers, GQA 32:16, 4608 Hidden Dim)
    pub fn gemma2_27b_it() -> Self {
        Self {
            architecture: ModelArchitecture::Gemma3,
            vocab_size: 256000,
            hidden_size: 4608,
            intermediate_size: 36864,
            num_hidden_layers: 46,
            num_attention_heads: 32,
            num_key_value_heads: 16,
            max_position_embeddings: 8192,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            moe: None,
            quantization: QuantizationMode::Q4_K_M,
        }
    }

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

    /// Compute exact per-layer Q4_K_M weight footprint in bytes (~4.85 bits/weight including block scales/mins)
    pub fn per_layer_weight_bytes_q4(&self) -> usize {
        let head_dim = self.hidden_size / self.num_attention_heads.max(1);
        let q_params = self.hidden_size * (self.num_attention_heads * head_dim);
        let kv_params = 2 * self.hidden_size * (self.num_key_value_heads * head_dim);
        let o_params = (self.num_attention_heads * head_dim) * self.hidden_size;
        let attn_params = q_params + kv_params + o_params;

        let ffn_single = 3 * self.hidden_size * self.intermediate_size;
        let ffn_params = if let Some(ref moe) = self.moe {
            ffn_single * (moe.num_routed_experts + moe.num_shared_experts).max(1)
        } else {
            ffn_single
        };

        // Q4_K_M averages ~4.85 bits/parameter (0.60625 bytes/parameter)
        ((attn_params + ffn_params) as f64 * 0.60625) as usize
    }

    /// Compute KV cache footprint in bytes per token across all layers for given K and V bit-widths
    pub fn kv_cache_bytes_per_token(&self, bits_k: usize, bits_v: usize) -> usize {
        let head_dim = self.hidden_size / self.num_attention_heads.max(1);
        let kv_dim = self.num_key_value_heads * head_dim;
        let total_bits = self.num_hidden_layers * kv_dim * (bits_k + bits_v);
        total_bits.div_ceil(8)
    }

    /// Configuration for Llama 3.3 70B Dense (80 Layers, 8192 Dim, 70.55B Parameters)
    pub fn llama3_70b() -> Self {
        Self {
            architecture: ModelArchitecture::Llama3,
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

    /// Configuration for Llama 4 Scout 109B Sparse MoE (109B Total, 17B Active Parameters)
    pub fn llama4_scout_109b() -> Self {
        Self {
            architecture: ModelArchitecture::Llama4,
            vocab_size: 128256,
            hidden_size: 5120,
            intermediate_size: 8192,
            num_hidden_layers: 48,
            num_attention_heads: 40,
            num_key_value_heads: 8,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            moe: Some(MoEConfig {
                num_routed_experts: 16,
                num_shared_experts: 1,
                num_active_experts: 2,
                routing_top_k: 2,
                norm_topk_prob: true,
            }),
            quantization: QuantizationMode::Q4_K_M,
        }
    }

    /// Backwards compatible alias for 70B Dense Architecture
    pub fn llama4_70b() -> Self {
        Self::llama3_70b()
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

    /// Configuration for Llama 3.3 70B Instruct
    pub fn llama3_3_70b() -> Self {
        Self {
            architecture: ModelArchitecture::Llama3,
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

    /// Configuration for Qwen 2.5 72B Instruct
    pub fn qwen2_5_72b() -> Self {
        Self {
            architecture: ModelArchitecture::Qwen2_5,
            vocab_size: 152064,
            hidden_size: 8192,
            intermediate_size: 29568,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-6,
            rope_theta: 1000000.0,
            moe: None,
            quantization: QuantizationMode::Q4_K_M,
        }
    }

    /// Configuration for DeepSeek-R1 / V3 Sparse MoE (671B Total, 37B Active)
    /// Ideal for 128GB+ Mac Unified Memory or multi-GPU sparse offloading
    pub fn deepseek_r1_671b() -> Self {
        Self {
            architecture: ModelArchitecture::DeepSeekV3,
            vocab_size: 129280,
            hidden_size: 7168,
            intermediate_size: 18432,
            num_hidden_layers: 61,
            num_attention_heads: 128,
            num_key_value_heads: 128,
            max_position_embeddings: 163840,
            rms_norm_eps: 1e-6,
            rope_theta: 10000.0,
            moe: Some(MoEConfig {
                num_routed_experts: 256,
                num_shared_experts: 1,
                num_active_experts: 8,
                routing_top_k: 8,
                norm_topk_prob: true,
            }),
            quantization: QuantizationMode::FP8,
        }
    }

    /// Configuration for Mixtral 8x22B Sparse MoE (141B Total, 39B Active)
    pub fn mixtral_8x22b() -> Self {
        Self {
            architecture: ModelArchitecture::Mistral,
            vocab_size: 32768,
            hidden_size: 6144,
            intermediate_size: 16384,
            num_hidden_layers: 56,
            num_attention_heads: 48,
            num_key_value_heads: 8,
            max_position_embeddings: 65536,
            rms_norm_eps: 1e-5,
            rope_theta: 1000000.0,
            moe: Some(MoEConfig {
                num_routed_experts: 8,
                num_shared_experts: 0,
                num_active_experts: 2,
                routing_top_k: 2,
                norm_topk_prob: false,
            }),
            quantization: QuantizationMode::Q4_K_M,
        }
    }

    /// Configuration for Phi-4 14B High-Reasoning System-1
    pub fn phi4_14b() -> Self {
        Self {
            architecture: ModelArchitecture::Phi4,
            vocab_size: 100352,
            hidden_size: 5120,
            intermediate_size: 17920,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: 10,
            max_position_embeddings: 16384,
            rms_norm_eps: 1e-5,
            rope_theta: 250000.0,
            moe: None,
            quantization: QuantizationMode::Q4_K_M,
        }
    }
}
