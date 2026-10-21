use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::{json, Value};

use crate::routes::chat::openai_error;
use crate::state::AppState;

/// `GET /v1/models`
///
/// Lists **only models the inference backend actually serves right now** (llama-server `--alias`
/// names, Ollama tags, ...). Each entry is the backend's own object, so fields such as
/// `meta.n_ctx_train` reach IDE clients unchanged. Nothing is synthesized: a model that is not
/// loaded is not advertised.
pub async fn list_models(State(state): State<AppState>) -> Response {
    let url = format!("{}/v1/models", state.backend.config.base_url.trim_end_matches('/'));
    let mut req = state.backend.client.get(&url);
    if let Some(ref k) = state.backend.config.api_key {
        req = req.header("Authorization", format!("Bearer {}", k));
    }

    let resp = match req.send().await {
        Ok(r) if r.status().is_success() => r,
        Ok(r) => {
            return openai_error(
                StatusCode::BAD_GATEWAY,
                format!("Backend /v1/models returned HTTP {}", r.status()),
                "bad_gateway",
            )
        }
        Err(e) => {
            return openai_error(
                StatusCode::SERVICE_UNAVAILABLE,
                format!(
                    "No inference backend reachable at {} ({}). Start it with ./launch_qwen32b_coder.ps1",
                    state.backend.config.base_url, e
                ),
                "backend_unavailable",
            )
        }
    };

    let parsed: Value = resp.json().await.unwrap_or_else(|_| json!({}));
    let data: Vec<Value> = parsed["data"].as_array().cloned().unwrap_or_default();

    Json(json!({
        "object": "list",
        "data": data,
    }))
    .into_response()
}

/// `GET /props` ? llama-server runtime properties (notably `default_generation_settings.n_ctx`,
/// the context the engine was actually launched with). IDE clients use it to size prompts.
pub async fn props(State(state): State<AppState>) -> Response {
    let url = format!("{}/props", state.backend.config.base_url.trim_end_matches('/'));
    match state.backend.client.get(&url).send().await {
        Ok(r) if r.status().is_success() => {
            let v: Value = r.json().await.unwrap_or_else(|_| json!({}));
            Json(v).into_response()
        }
        Ok(r) => openai_error(StatusCode::BAD_GATEWAY, format!("Backend /props returned HTTP {}", r.status()), "bad_gateway"),
        Err(e) => openai_error(
            StatusCode::SERVICE_UNAVAILABLE,
            format!("No inference backend reachable at {} ({})", state.backend.config.base_url, e),
            "backend_unavailable",
        ),
    }
}
