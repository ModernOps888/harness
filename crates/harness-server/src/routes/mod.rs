pub mod chat;
pub mod metrics;
pub mod models;

use axum::extract::DefaultBodyLimit;
use axum::http::HeaderValue;
use axum::routing::{get, post};
use axum::Router;
use tower_http::cors::{AllowOrigin, Any, CorsLayer};
use crate::state::AppState;

/// Only loopback web origins may call the API from a browser (the server has no auth).
/// The host is parsed and compared EXACTLY: `http://localhost.evil.com` must not pass.
pub fn is_loopback_origin(origin: &HeaderValue) -> bool {
    let o = origin.to_str().unwrap_or("");
    let rest = match o.strip_prefix("http://").or_else(|| o.strip_prefix("https://")) {
        Some(r) => r,
        None => return false,
    };
    let authority = rest.split('/').next().unwrap_or("");
    let host = if let Some(stripped) = authority.strip_prefix('[') {
        // IPv6 literal: [::1]:3000
        stripped.split(']').next().unwrap_or("")
    } else {
        authority.split(':').next().unwrap_or("")
    };
    matches!(host, "localhost" | "127.0.0.1" | "::1")
}

pub fn create_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(AllowOrigin::predicate(|origin, _| is_loopback_origin(origin)))
        .allow_methods(Any)
        .allow_headers(Any);

    Router::new()
        .route("/health", get(metrics::health_check))
        .route("/report", get(metrics::report_handler))
        .route("/proof", get(metrics::proof_handler))
        .route("/proofs", get(metrics::proof_handler))
        .route("/metrics", get(metrics::metrics_handler))
        .route("/v1/models", get(models::list_models))
        .route("/props", get(models::props))
        .route("/v1/chat/completions", post(chat::chat_completions))
        .route("/v1/completions", post(chat::completions))
        .route("/v1/embeddings", post(chat::embeddings))
        .route("/infill", post(chat::infill))
        .route("/tokenize", post(chat::tokenize))
        .route("/apply-template", post(chat::apply_template))
        .layer(cors)
        .layer(DefaultBodyLimit::max(16 * 1024 * 1024))
        .with_state(state)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ok(o: &str) -> bool {
        is_loopback_origin(&HeaderValue::from_str(o).unwrap())
    }

    #[test]
    fn loopback_origins_accepted() {
        assert!(ok("http://localhost:3000"));
        assert!(ok("http://127.0.0.1:5173"));
        assert!(ok("https://localhost"));
        assert!(ok("http://[::1]:8090"));
    }

    #[test]
    fn lookalike_and_foreign_origins_rejected() {
        assert!(!ok("http://localhost.evil.com"));
        assert!(!ok("http://127.0.0.1.evil.com:80"));
        assert!(!ok("http://evil.com"));
        assert!(!ok("http://evil.com/localhost"));
        assert!(!ok("vscode-webview://abc"));
        assert!(!ok("null"));
        assert!(!ok("http://localhostx"));
    }
}
