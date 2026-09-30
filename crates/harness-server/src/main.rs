use harness_server::routes::create_router;
use harness_server::AppState;
use std::net::SocketAddr;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Initializing HARNESS Pure-Rust LLM Inference Engine...");

    let state = AppState::new();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = create_router(state).layer(cors);

    let addr = SocketAddr::from(([0, 0, 0, 0], 8080));
    info!("🚀 HARNESS Server listening on http://{}", addr);
    info!("🔗 OpenAI API endpoint: http://{}/v1/chat/completions", addr);
    info!("📊 Engine metrics: http://{}/metrics", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
