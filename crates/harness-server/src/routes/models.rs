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
}

#[derive(Debug, Serialize)]
pub struct ModelListResponse {
    pub object: String,
    pub data: Vec<ModelCard>,
}

pub async fn list_models(State(state): State<AppState>) -> Json<ModelListResponse> {
    let current_name = state.model_name.read().unwrap().clone();
    let config = state.model_config.read().unwrap().clone();

    let models = vec![
        ModelCard {
            id: current_name,
            object: "model".into(),
            created: 1727654400,
            owned_by: "harness".into(),
            architecture: format!("{:?}", config.architecture),
            quantization: format!("{:?}", config.quantization),
            context_length: config.max_position_embeddings,
        },
        ModelCard {
            id: "Llama-4-Scout-70B-LayerStream".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "harness".into(),
            architecture: "Llama4".into(),
            quantization: "Q4_K_M".into(),
            context_length: 131072,
        },
        ModelCard {
            id: "DeepSeek-V4.1-Flash-MoE".into(),
            object: "model".into(),
            created: 1727654400,
            owned_by: "harness".into(),
            architecture: "DeepSeekV4".into(),
            quantization: "FP8".into(),
            context_length: 1048576,
        },
    ];

    Json(ModelListResponse {
        object: "list".into(),
        data: models,
    })
}
