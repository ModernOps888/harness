use harness_core::{MoEConfig, ModelArchitecture, ModelConfig, QuantizationMode, Result};
use std::fs::File;
use std::path::Path;

pub struct ConfigLoader;

impl ConfigLoader {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<ModelConfig> {
        let file = File::open(path)?;
        let val: serde_json::Value = serde_json::from_reader(file)?;

        let model_type = val.get("model_type").and_then(|v| v.as_str()).unwrap_or("llama");
        let arch = match model_type {
            "qwen2" => ModelArchitecture::Qwen2_5,
            "qwen" | "qwen3" | "qwen3_moe" => ModelArchitecture::Qwen3,
            "deepseek_v2" | "deepseek_v3" => ModelArchitecture::DeepSeekV3,
            "deepseek" | "deepseek_v4" => ModelArchitecture::DeepSeekV4,
            "phi3" | "phi4" => ModelArchitecture::Phi4,
            "gemma" | "gemma2" | "gemma3" => ModelArchitecture::Gemma3,
            "mistral" => ModelArchitecture::Mistral,
            _ => ModelArchitecture::Llama4,
        };

        let vocab_size = val.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(128256) as usize;
        let hidden_size = val.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(4096) as usize;
        let intermediate_size = val
            .get("moe_intermediate_size")
            .or_else(|| val.get("intermediate_size"))
            .and_then(|v| v.as_u64())
            .unwrap_or(11008) as usize;
        let num_hidden_layers = val.get("num_hidden_layers").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let num_attention_heads = val.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(32) as usize;
        let num_key_value_heads = val.get("num_key_value_heads").and_then(|v| v.as_u64()).unwrap_or(8) as usize;
        let max_position_embeddings = val.get("max_position_embeddings").and_then(|v| v.as_u64()).unwrap_or(8192) as usize;
        let rms_norm_eps = val.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-5) as f32;
        let rope_theta = val.get("rope_theta").and_then(|v| v.as_f64()).unwrap_or(500000.0) as f32;

        let num_routed = val
            .get("num_experts")
            .or_else(|| val.get("n_routed_experts"))
            .or_else(|| val.get("num_local_experts"))
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as usize;

        let moe = if num_routed > 0 {
            let top_k = val
                .get("num_experts_per_tok")
                .or_else(|| val.get("num_selected_experts"))
                .and_then(|v| v.as_u64())
                .unwrap_or(2) as usize;
            let shared = val
                .get("n_shared_experts")
                .or_else(|| val.get("num_shared_experts"))
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as usize;
            let norm_topk_prob = val
                .get("norm_topk_prob")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            Some(MoEConfig {
                num_routed_experts: num_routed,
                num_shared_experts: shared,
                num_active_experts: top_k,
                routing_top_k: top_k,
                norm_topk_prob,
            })
        } else {
            None
        };

        Ok(ModelConfig {
            architecture: arch,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            moe,
            quantization: QuantizationMode::None,
        })
    }
}
