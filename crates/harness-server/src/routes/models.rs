use axum::extract::State;
use axum::Json;
use serde::Serialize;
use crate::state::AppState;

#[derive(Debug, Serialize)]
pub struct ModelCard {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
    pub architecture: String,
    pub quantization: String,
    pub context_length: usize,
    pub hardware_tier: String,
    pub sparse_routing: String,
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let current_name = state.model_name.read().unwrap().clone();
    let config = state.model_config.read().unwrap().clone();

    let backend_models: Vec<String> = state.backend.list_models().await;
    let mut models = Vec::new();

    for m in backend_models {
        models.push(ModelCard {
            id: m.clone(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "local-backend".into(),
            architecture: "Transformer".into(),
            quantization: "Native".into(),
            context_length: config.max_position_embeddings,
            hardware_tier: "Active Hardware Accelerated".into(),
            sparse_routing: "Direct Execution".into(),
        });
    }

    if !models.iter().any(|m| m.id == current_name) {
        models.push(ModelCard {
            id: current_name,
            object: "model".into(),
            created: 1727654400,
            owned_by: "harness".into(),
            architecture: format!("{:?}", config.architecture),
            quantization: format!("{:?}", config.quantization),
            context_length: config.max_position_embeddings,
            hardware_tier: "Tier-2: 16GB GPU / 32GB RAM resident".into(),
            sparse_routing: "LIF Spiking Sparsity (adaptive thresholding)".into(),
        });
    }

    models.extend(vec![
        ModelCard {
            id: "Llama-3.3-70B-LayerStream".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "harness".into(),
            architecture: "Llama3".into(),
            quantization: "Q4_K_M".into(),
            context_length: 131072,
            hardware_tier: "Tier-1: 8GB GPU (PCIe Ping-Pong DMA) | Tier-3: 64GB-128GB Mac UMA fully resident".into(),
            sparse_routing: "Dense layer streaming with double-buffered VRAM staging".into(),
        },
        ModelCard {
            id: "Qwen-2.5-72B-Instruct".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "harness".into(),
            architecture: "Qwen2_5".into(),
            quantization: "Q4_K_M".into(),
            context_length: 131072,
            hardware_tier: "Tier-3: 32GB GPU or 64GB-128GB Mac UMA (up to 3x concurrent instances on 128GB)".into(),
            sparse_routing: "Rotary RoPE + FlashAttention v3 multi-core".into(),
        },
        ModelCard {
            id: "DeepSeek-R1-671B-SparseMoE".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "harness".into(),
            architecture: "DeepSeekV3".into(),
            quantization: "FP8 / 2-bit Ternary".into(),
            context_length: 163840,
            hardware_tier: "Tier-3+: 128GB-192GB Mac UMA or 32GB GPU with expert offload".into(),
            sparse_routing: "256 routed experts (8 active = 37B active, 4x compute speedup via activation sparsity)".into(),
        },
        ModelCard {
            id: "Mixtral-8x22B-SparseMoE".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "harness".into(),
            architecture: "Mistral".into(),
            quantization: "Q4_K_M".into(),
            context_length: 65536,
            hardware_tier: "Tier-3: 24GB-32GB GPU or 64GB-96GB Mac UMA resident".into(),
            sparse_routing: "8 experts (2 active = 39B active, 4x FLOP efficiency)".into(),
        },
        ModelCard {
            id: "Phi-4-14B-Reasoning".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "harness".into(),
            architecture: "Phi4".into(),
            quantization: "Q4_K_M".into(),
            context_length: 16384,
            hardware_tier: "Tier-1: 8GB GPU / 16GB Mac UMA fully resident".into(),
            sparse_routing: "Dense System-1 fast intuitive inference".into(),
        },
    ]);

    Json(ModelListResponse {
        object: "list".into(),
        data: models,
    })
}
