# ══════════════════════════════════════════════════════════════════════
#  HARNESS PURE-RUST LLM INFERENCE ORCHESTRATION & SAFETY MIDDLEWARE
#  Multi-Stage Container: Ubuntu 24.04 + Rust 1.85+ + Node.js 22 + CUDA/Metal/ROCm
# ══════════════════════════════════════════════════════════════════════

# ── Stage 1: Frontend Build ──────────────────────────────────────────
FROM node:22-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Rust Backend Build ──────────────────────────────────────
FROM rust:1.85-bookworm AS backend-builder
WORKDIR /app

# Install build essentials and CUDA development libraries if present
RUN apt-get update && apt-get install -y --no-install-recommends \
    pkg-config \
    libssl-dev \
    cmake \
    clang \
    && rm -rf /var/lib/apt/lists/*

COPY Cargo.toml Cargo.lock ./
COPY crates/ ./crates/
COPY tests/ ./tests/

# Compile production binary with native optimizations
RUN cargo build --release -p harness-cli

# ── Stage 3: Minimal Production Runtime ──────────────────────────────
FROM debian:bookworm-slim AS runtime
WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    libssl3 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 10001 -m harness

# Copy backend compiled executable
COPY --from=backend-builder /app/target/release/harness-cli /usr/local/bin/harness

# Copy frontend distribution
COPY --from=frontend-builder /app/frontend/dist /app/static

RUN chown -R harness:harness /app

ENV RUST_LOG=info
ENV HARNESS_HOST=0.0.0.0
ENV HARNESS_PORT=8080

USER harness

EXPOSE 8080

# Auto-detect hardware on startup and launch server
ENTRYPOINT ["harness"]
CMD ["serve", "--port", "8080"]
