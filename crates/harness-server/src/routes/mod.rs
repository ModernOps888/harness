pub mod chat;
pub mod metrics;
pub mod models;

use axum::routing::{get, post};
use axum::Router;
use crate::state::AppState;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        .route("/health", get(metrics::health_check))
        .route("/metrics", get(metrics::metrics_handler))
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .with_state(state)
}
