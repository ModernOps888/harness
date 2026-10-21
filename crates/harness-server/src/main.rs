use harness_server::routes::create_router;
use harness_server::AppState;
use std::net::SocketAddr;
use tracing::info;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .with(tracing_subscriber::fmt::layer())
        .init();

    info!("Initializing HARNESS Pure-Rust LLM Inference Orchestration Middleware...");

    let state = AppState::new();

    // CORS is configured (loopback-only) inside `create_router`.
    let app = create_router(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 8090));
    info!("🚀 HARNESS Server listening on http://{}", addr);
    info!("🔗 OpenAI API endpoint: http://{}/v1/chat/completions", addr);
    info!("📊 Telemetry & safety metrics: http://{}/metrics", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
