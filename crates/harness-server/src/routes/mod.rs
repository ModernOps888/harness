pub mod chat;
pub mod metrics;
pub mod models;

use axum::extract::DefaultBodyLimit;
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{Any, CorsLayer};
use crate::state::AppState;

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(metrics::health_check))
        .route("/report", get(metrics::report_handler))
        .route("/proof", get(metrics::proof_handler))
        .route("/proofs", get(metrics::proof_handler))
        .route("/metrics", get(metrics::metrics_handler))
        .route("/v1/models", get(models::list_models))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .layer(cors)
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(state)
}
