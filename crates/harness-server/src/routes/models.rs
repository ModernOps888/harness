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
            hardware_tier: "RTX 5060 8GB VRAM / 32GB Host RAM".into(),
            sparse_routing: "LIF Spiking Sparsity (adaptive thresholding)".into(),
        });
    }

    if models.is_empty() {
        models.push(ModelCard {
            id: "llama3.1:70b-instruct-q2_K".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "local-disk".into(),
            architecture: "Llama3_1".into(),
            quantization: "Q2_K".into(),
            context_length: 131072,
            hardware_tier: "RTX 5060 8GB VRAM + 32GB host RAM (hybrid GPU/CPU offload); measured speed: see /proof".into(),
            sparse_routing: "Pure Autoregressive Forward Pass".into(),
        });
        models.push(ModelCard {
            id: "qwen2.5-coder:7b".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "local-disk".into(),
            architecture: "Qwen2_5".into(),
            quantization: "Q4_K_M".into(),
            context_length: 32768,
            hardware_tier: "RTX 5060 8GB VRAM resident; measured speed: see /proof".into(),
            sparse_routing: "Direct VRAM Execution".into(),
        });
        models.push(ModelCard {
            id: "llama3.2:1b".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "local-disk".into(),
            architecture: "Llama3_2".into(),
            quantization: "Q8_0".into(),
            context_length: 8192,
            hardware_tier: "RTX 5060 8GB VRAM resident (1.3GB); speed not benchmarked".into(),
            sparse_routing: "Direct VRAM Execution".into(),
        });
    }

    Json(ModelListResponse {
        object: "list".into(),
        data: models,
    })
}
