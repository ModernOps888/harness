//! Real LLM Backend Proxy
//!
//! Connects HARNESS to actual LLM inference backends (Ollama, llama.cpp server,
//! vLLM, or any OpenAI-compatible API). HARNESS applies its enhancements
//! (lateral inhibition, entropy scoring, DFA masking) on top of real model output.

use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Supported backend types
#[derive(Debug, Clone)]
pub enum BackendType {
    /// Ollama at http://localhost:11434 (default)
    Ollama,
    /// Any OpenAI-compatible API (llama.cpp server, vLLM, LM Studio, etc.)
    OpenAICompatible,
}

/// Configuration for connecting to a real LLM backend
#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub base_url: String,
    pub backend_type: BackendType,
    pub timeout_secs: u64,
    pub api_key: Option<String>,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            base_url: "http://localhost:11434".to_string(),
            backend_type: BackendType::Ollama,
            timeout_secs: 600,
            api_key: None,
        }
    }
}

impl BackendConfig {
    /// Auto-detect available backends by probing known ports (prioritizing port 8089 32B Coder Speculative Server)
    pub async fn auto_detect() -> Self {
        let client = Client::builder()
            .timeout(Duration::from_secs(2))
            .build()
            .unwrap_or_default();

        // 1. Highest Priority: Dedicated HARNESS Qwen2.5-Coder-32B + 1.5B Speculative llama-server (port 8089)
        if let Ok(resp) = client.get("http://127.0.0.1:8089/health").send().await {
            if resp.status().is_success() {
                return Self {
                    base_url: "http://127.0.0.1:8089".to_string(),
                    backend_type: BackendType::OpenAICompatible,
                    timeout_secs: 600,
                    api_key: None,
                };
            }
        }
        if let Ok(resp) = client.get("http://127.0.0.1:8089/v1/models").send().await {
            if resp.status().is_success() {
                return Self {
                    base_url: "http://127.0.0.1:8089".to_string(),
                    backend_type: BackendType::OpenAICompatible,
                    timeout_secs: 600,
                    api_key: None,
                };
            }
        }

        // 2. Try Ollama (port 11434)
        if let Ok(resp) = client.get("http://localhost:11434/api/tags").send().await {
            if resp.status().is_success() {
                return Self {
                    base_url: "http://localhost:11434".to_string(),
                    backend_type: BackendType::Ollama,
                    timeout_secs: 600,
                    api_key: None,
                };
            }
        }

        // 3. Try llama.cpp server / LM Studio (port 1234)
        if let Ok(resp) = client.get("http://localhost:1234/v1/models").send().await {
            if resp.status().is_success() {
                return Self {
                    base_url: "http://localhost:1234".to_string(),
                    backend_type: BackendType::OpenAICompatible,
                    timeout_secs: 600,
                    api_key: None,
                };
            }
        }

        // 4. Try vLLM / generic (port 8000)
        if let Ok(resp) = client.get("http://localhost:8000/v1/models").send().await {
            if resp.status().is_success() {
                return Self {
                    base_url: "http://localhost:8000".to_string(),
                    backend_type: BackendType::OpenAICompatible,
                    timeout_secs: 600,
                    api_key: None,
                };
            }
        }

        // Default: Ollama (user may start it later)
        Self::default()
    }

    /// Check if backend is currently reachable
    pub async fn is_available(&self) -> bool {
        let client = Client::builder()
            .timeout(Duration::from_secs(3))
            .build()
            .unwrap_or_default();

        match self.backend_type {
            BackendType::Ollama => {
                let url = format!("{}/api/tags", self.base_url);
                client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
            }
            BackendType::OpenAICompatible => {
                let models_url = format!("{}/v1/models", self.base_url);
                if client.get(&models_url).send().await.map(|r| r.status().is_success()).unwrap_or(false) {
                    return true;
                }
                let health_url = format!("{}/health", self.base_url);
                client.get(&health_url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
            }
        }
    }

    /// Get the chat completions endpoint URL
    pub fn chat_url(&self) -> String {
        match self.backend_type {
            BackendType::Ollama => format!("{}/api/chat", self.base_url),
            BackendType::OpenAICompatible => format!("{}/v1/chat/completions", self.base_url),
        }
    }

    /// Map a HARNESS model name to the backend's model identifier (prioritizing 27B-35B Coder models)
    pub fn resolve_model_name(&self, harness_model: &str) -> String {
        match self.backend_type {
            BackendType::Ollama => {
                let lower = harness_model.trim().to_lowercase();
                match harness_model {
                    _ if lower.is_empty() || lower == "default" || lower == "auto" => {
                        "qwen2.5-coder:32b".to_string()
                    }
                    _ if lower.contains("32b") || lower.contains("27b") || lower.contains("qwen3") => {
                        "qwen2.5-coder:32b".to_string()
                    }
                    _ if lower.contains("33b") => "deepseek-coder:33b".to_string(),
                    _ if lower.contains("30b") || lower.contains("a3b") => "qwen3:30b-a3b".to_string(),
                    _ if lower.contains("14b") => "qwen2.5-coder:14b-instruct-q4_K_S".to_string(),
                    _ if lower.contains("1.5b") => "qwen2.5-coder:1.5b".to_string(),
                    _ if lower.contains("7b") => "qwen2.5-coder:7b".to_string(),
                    _ if lower.contains("coder") || lower.contains("code") => "qwen2.5-coder:32b".to_string(),
                    m if m.contains("671B") || lower.contains("671b") => "deepseek-r1:671b".to_string(),
                    m if m.contains("70B") || lower.contains("70b") || m.contains("Scout") => {
                        "llama3.1:70b-instruct-q2_K".to_string()
                    }
                    m if m.contains("72B") || lower.contains("72b") => "qwen2.5:72b-instruct-q4_K_M".to_string(),
                    m if m.contains("Phi") => "phi4:14b".to_string(),
                    m if m.contains("Mixtral") => "mixtral:8x22b".to_string(),
                    m if m.contains("Flash") => "deepseek-r1:8b".to_string(),
                    _ => harness_model.to_lowercase().replace(' ', ":"),
                }
            }
            BackendType::OpenAICompatible => {
                let lower = harness_model.trim().to_lowercase();
                if lower.is_empty() || lower == "default" || lower == "auto" {
                    "qwen2.5-coder:32b".to_string()
                } else {
                    harness_model.to_string()
                }
            }
        }
    }
}

// ---- Ollama-native request/response types ----

#[derive(Debug, Serialize)]
pub struct OllamaChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub keep_alive: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct OllamaChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Serialize)]
pub struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_predict: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_gpu: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct OllamaChatResponse {
    pub model: Option<String>,
    pub message: Option<OllamaChatMessage>,
    pub done: Option<bool>,
    pub total_duration: Option<u64>,
    pub eval_count: Option<u64>,
    pub eval_duration: Option<u64>,
    pub prompt_eval_count: Option<u64>,
    pub prompt_eval_duration: Option<u64>,
}

// ---- OpenAI-compatible request/response types ----

#[derive(Debug, Serialize)]
pub struct OpenAIChatRequest {
    pub model: String,
    pub messages: Vec<OllamaChatMessage>,
    pub stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<usize>,
}

/// The real LLM backend proxy client
pub struct BackendProxy {
    pub config: BackendConfig,
    pub client: Client,
}

#[derive(Debug, Deserialize)]
pub struct OllamaTagsResponse {
    pub models: Vec<OllamaModelTag>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OllamaModelTag {
    pub name: String,
    pub size: u64,
}

#[derive(Debug, Deserialize)]
pub struct OllamaPsResponse {
    pub models: Vec<OllamaPsModel>,
}

#[derive(Debug, Deserialize, Clone)]
pub struct OllamaPsModel {
    pub name: String,
    pub model: String,
    pub size_vram: Option<u64>,
}

impl BackendProxy {
    pub fn new(config: BackendConfig) -> Self {
        // No *total* request timeout: a streamed generation (after a multi-minute prefill) can
        // legitimately run for many minutes. Bound connect and inter-byte silence instead; the
        // silence bound must exceed the longest prefill (llama-server sends nothing meanwhile).
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(config.timeout_secs.max(1800)))
            .build()
            .unwrap_or_default();

        Self { config, client }
    }

    /// Forward an OpenAI-format JSON body verbatim to the backend's `/v1/<path>` endpoint.
    ///
    /// Both llama.cpp `llama-server` and Ollama expose the OpenAI wire format under `/v1/`, so
    /// this is the single lossless path used for chat (incl. `tools` / `tool_calls`), FIM
    /// completions and embeddings. Nothing in `body` is rewritten except what the caller
    /// already changed (the resolved model id).
    pub async fn forward_json(
        &self,
        path: &str,
        body: &serde_json::Value,
    ) -> Result<reqwest::Response, BackendError> {
        let url = format!("{}/{}", self.config.base_url.trim_end_matches('/'), path.trim_start_matches('/'));
        let mut request = self.client.post(url).json(body);
        if let Some(ref key) = self.config.api_key {
            request = request.header("Authorization", format!("Bearer {}", key));
        }
        request
            .send()
            .await
            .map_err(|e| BackendError::ConnectionFailed(e.to_string()))
    }

    /// List models currently loaded and resident in memory on the backend
    pub async fn running_models(&self) -> Vec<String> {
        match self.config.backend_type {
            BackendType::Ollama => {
                let url = format!("{}/api/ps", self.config.base_url);
                if let Ok(resp) = self.client.get(&url).send().await {
                    if let Ok(ps) = resp.json::<OllamaPsResponse>().await {
                        return ps.models.into_iter().map(|m| m.name).collect();
                    }
                }
                vec![]
            }
            BackendType::OpenAICompatible => vec![],
        }
    }

    /// List models currently available on the active backend
    pub async fn list_models(&self) -> Vec<String> {
        match self.config.backend_type {
            BackendType::Ollama => {
                let url = format!("{}/api/tags", self.config.base_url);
                if let Ok(resp) = self.client.get(&url).send().await {
                    if let Ok(tags) = resp.json::<OllamaTagsResponse>().await {
                        return tags.models.into_iter().map(|m| m.name).collect();
                    }
                }
                vec![]
            }
            BackendType::OpenAICompatible => {
                let url = format!("{}/v1/models", self.config.base_url);
                let mut req = self.client.get(&url);
                if let Some(ref k) = self.config.api_key {
                    req = req.header("Authorization", format!("Bearer {}", k));
                }
                if let Ok(resp) = req.send().await {
                    if let Ok(json) = resp.json::<serde_json::Value>().await {
                        if let Some(data) = json["data"].as_array() {
                            return data.iter()
                                .filter_map(|m| m["id"].as_str().map(|s| s.to_string()))
                                .collect();
                        }
                    }
                }
                vec![]
            }
        }
    }

    /// Resolve requested model name against actual models installed on backend (prioritizing 27B-35B Coder models)
    pub async fn resolve_model_smart(&self, requested: &str) -> String {
        let req_clean = requested.trim().to_lowercase();

        // 0. Active Resident Memory Check:
        // If a model is ALREADY resident and active in memory (e.g. 32B Coder loaded),
        // and the user requested a generic placeholder ("qwen2.5-coder:32b", "default", "", "auto")
        // or a model keyword matching what is already warm, REUSE IT immediately to prevent eviction thrashing!
        let running = self.running_models().await;
        if let Some(resident) = running.first() {
            let res_clean = resident.to_lowercase();
            if req_clean.is_empty()
                || req_clean == "default"
                || req_clean == "auto"
                || (req_clean.contains("32b") && res_clean.contains("32b"))
                || (req_clean.contains("27b") && (res_clean.contains("27b") || res_clean.contains("32b")))
                || (req_clean.contains("70b") && res_clean.contains("70b"))
                || (req_clean.contains("llama") && res_clean.contains("llama"))
                || (req_clean.contains("qwen") && res_clean.contains("qwen") && !res_clean.contains("1.5b"))
            {
                return resident.clone();
            }
        }

        let available = self.list_models().await;
        if available.is_empty() {
            return self.config.resolve_model_name(requested);
        }

        // 1. Direct match
        if let Some(exact) = available.iter().find(|m| {
            let ml = m.to_lowercase();
            ml == req_clean || ml.starts_with(&format!("{}:", req_clean))
        }) {
            return exact.clone();
        }

        // 2. Keyword matching (27B-35B Coder Tier prioritized first!)
        if req_clean.is_empty()
            || req_clean == "default"
            || req_clean == "auto"
            || req_clean.contains("32b")
            || req_clean.contains("27b")
            || req_clean.contains("33b")
            || req_clean.contains("35b")
            || req_clean.contains("coder")
            || req_clean.contains("code")
        {
            // Explicit 14B or 7B or 1.5B check before defaulting to 32B
            if req_clean.contains("14b") {
                if let Some(m14) = available.iter().find(|m| m.to_lowercase().contains("14b")) {
                    return m14.clone();
                }
            }
            if req_clean.contains("7b") {
                if let Some(m7) = available.iter().find(|m| m.to_lowercase().contains("7b")) {
                    return m7.clone();
                }
            }
            if req_clean.contains("1.5b") {
                if let Some(m15) = available.iter().find(|m| m.to_lowercase().contains("1.5b")) {
                    return m15.clone();
                }
            }
            // Prefer 32B / 33B / 27B / 30B Coder first (never accidentally pick 1.5b alphabetically!)
            if let Some(m32) = available.iter().find(|m| {
                let ml = m.to_lowercase();
                (ml.contains("32b") || ml.contains("33b") || ml.contains("27b") || ml.contains("30b"))
                    && (ml.contains("coder") || ml.contains("qwen") || ml.contains("deepseek"))
            }) {
                return m32.clone();
            }
            if let Some(m32_any) = available.iter().find(|m| {
                let ml = m.to_lowercase();
                ml.contains("32b") || ml.contains("33b") || ml.contains("27b") || ml.contains("30b")
            }) {
                return m32_any.clone();
            }
            if let Some(m14) = available.iter().find(|m| {
                let ml = m.to_lowercase();
                ml.contains("coder") && ml.contains("14b")
            }) {
                return m14.clone();
            }
            if let Some(m7) = available.iter().find(|m| {
                let ml = m.to_lowercase();
                ml.contains("coder") && !ml.contains("1.5b")
            }) {
                return m7.clone();
            }
        }
        if req_clean.contains("70b") || req_clean.contains("72b") || req_clean.contains("scout") {
            if let Some(m70) = available.iter().find(|m| m.contains("70b") || m.contains("72b")) {
                return m70.clone();
            }
        }
        if req_clean.contains("14b") {
            if let Some(m14) = available.iter().find(|m| m.to_lowercase().contains("14b")) {
                return m14.clone();
            }
        }
        if req_clean.contains("7b") || req_clean.contains("8b") {
            if let Some(m7) = available.iter().find(|m| m.contains("7b") || m.contains("8b")) {
                return m7.clone();
            }
        }
        if req_clean.contains("qwen") {
            if let Some(qwen) = available.iter().find(|m| {
                let ml = m.to_lowercase();
                ml.contains("qwen") && !ml.contains("1.5b")
            }) {
                return qwen.clone();
            }
        }
        if req_clean.contains("llama") {
            if let Some(llama) = available.iter().find(|m| m.contains("llama")) {
                return llama.clone();
            }
        }

        // 3. Fallback to currently running model before any cold model
        if let Some(resident) = running.first() {
            return resident.clone();
        }

        // 4. First available non-draft model
        if let Some(first_non_draft) = available.iter().find(|m| !m.to_lowercase().contains("1.5b") && !m.to_lowercase().contains("1b")) {
            return first_non_draft.clone();
        }
        if let Some(first) = available.first() {
            return first.clone();
        }

        self.config.resolve_model_name(requested)
    }

    /// Send a non-streaming chat request to the real backend and get the full response
    pub async fn chat_completion(
        &self,
        model: &str,
        messages: &[(String, String)], // (role, content) pairs
        temperature: f32,
        max_tokens: Option<usize>,
    ) -> Result<(String, BackendMetrics), BackendError> {
        let resolved_model = self.resolve_model_smart(model).await;
        let chat_messages: Vec<OllamaChatMessage> = messages
            .iter()
            .map(|(role, content)| OllamaChatMessage {
                role: role.clone(),
                content: content.clone(),
            })
            .collect();

        match self.config.backend_type {
            BackendType::Ollama => {
                let req_body = OllamaChatRequest {
                    model: resolved_model,
                    messages: chat_messages,
                    stream: false,
                    keep_alive: Some("24h".to_string()),
                    options: Some(OllamaOptions {
                        temperature: Some(temperature),
                        top_p: None,
                        num_predict: max_tokens,
                        num_gpu: None,
                    }),
                };

                let resp = self.client
                    .post(self.config.chat_url())
                    .json(&req_body)
                    .send()
                    .await
                    .map_err(|e| BackendError::ConnectionFailed(e.to_string()))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(BackendError::RequestFailed(format!("HTTP {}: {}", status, body)));
                }

                let ollama_resp: OllamaChatResponse = resp.json().await
                    .map_err(|e| BackendError::ParseError(e.to_string()))?;

                let content = ollama_resp.message
                    .map(|m| m.content)
                    .unwrap_or_default();

                let prompt_eval_dur = ollama_resp.prompt_eval_duration.unwrap_or(0);
                let ttft_ms = if prompt_eval_dur > 0 {
                    prompt_eval_dur as f64 / 1_000_000.0
                } else {
                    0.0
                };
                let metrics = BackendMetrics {
                    eval_tokens: ollama_resp.eval_count.unwrap_or(0),
                    eval_duration_ns: ollama_resp.eval_duration.unwrap_or(0),
                    prompt_eval_tokens: ollama_resp.prompt_eval_count.unwrap_or(0),
                    prompt_eval_duration_ns: prompt_eval_dur,
                    tok_per_sec: if let (Some(count), Some(dur)) = (ollama_resp.eval_count, ollama_resp.eval_duration) {
                        if dur > 0 { (count as f64 / dur as f64) * 1_000_000_000.0 } else { 0.0 }
                    } else {
                        0.0
                    },
                    ttft_ms,
                };

                Ok((content, metrics))
            }
            BackendType::OpenAICompatible => {
                let req_body = OpenAIChatRequest {
                    model: resolved_model,
                    messages: chat_messages,
                    stream: false,
                    temperature: Some(temperature),
                    top_p: None,
                    max_tokens,
                };

                let mut request = self.client.post(self.config.chat_url()).json(&req_body);
                if let Some(ref key) = self.config.api_key {
                    request = request.header("Authorization", format!("Bearer {}", key));
                }

                let resp = request.send().await
                    .map_err(|e| BackendError::ConnectionFailed(e.to_string()))?;

                if !resp.status().is_success() {
                    let status = resp.status();
                    let body = resp.text().await.unwrap_or_default();
                    return Err(BackendError::RequestFailed(format!("HTTP {}: {}", status, body)));
                }

                let json_resp: serde_json::Value = resp.json().await
                    .map_err(|e| BackendError::ParseError(e.to_string()))?;

                let content = json_resp["choices"][0]["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();

                let predicted_per_sec = json_resp["timings"]["predicted_per_second"]
                    .as_f64()
                    .unwrap_or(0.0);
                let prompt_ms = json_resp["timings"]["prompt_ms"]
                    .as_f64()
                    .unwrap_or(0.0);
                let predicted_ms = json_resp["timings"]["predicted_ms"]
                    .as_f64()
                    .unwrap_or(0.0);

                let metrics = BackendMetrics {
                    eval_tokens: json_resp["usage"]["completion_tokens"].as_u64().unwrap_or(0),
                    eval_duration_ns: (predicted_ms * 1_000_000.0) as u64,
                    prompt_eval_tokens: json_resp["usage"]["prompt_tokens"].as_u64().unwrap_or(0),
                    prompt_eval_duration_ns: (prompt_ms * 1_000_000.0) as u64,
                    tok_per_sec: predicted_per_sec,
                    ttft_ms: prompt_ms,
                };

                Ok((content, metrics))
            }
        }
    }

    /// Send a streaming chat request and return individual SSE chunks via a channel
    pub async fn chat_completion_stream(
        &self,
        model: &str,
        messages: &[(String, String)],
        temperature: f32,
        max_tokens: Option<usize>,
        tx: tokio::sync::mpsc::Sender<Result<StreamChunk, BackendError>>,
    ) {
        self.chat_completion_stream_with_gpu(model, messages, temperature, max_tokens, None, tx).await;
    }

    /// Send a streaming chat request with explicit GPU layer offloading control
    pub async fn chat_completion_stream_with_gpu(
        &self,
        model: &str,
        messages: &[(String, String)],
        temperature: f32,
        max_tokens: Option<usize>,
        num_gpu: Option<usize>,
        tx: tokio::sync::mpsc::Sender<Result<StreamChunk, BackendError>>,
    ) {
        let resolved_model = self.resolve_model_smart(model).await;
        let chat_messages: Vec<OllamaChatMessage> = messages
            .iter()
            .map(|(role, content)| OllamaChatMessage {
                role: role.clone(),
                content: content.clone(),
            })
            .collect();

        match self.config.backend_type {
            BackendType::Ollama => {
                let req_body = OllamaChatRequest {
                    model: resolved_model,
                    messages: chat_messages,
                    stream: true,
                    keep_alive: Some("24h".to_string()),
                    options: Some(OllamaOptions {
                        temperature: Some(temperature),
                        top_p: None,
                        num_predict: max_tokens,
                        num_gpu,
                    }),
                };

                let resp = match self.client
                    .post(self.config.chat_url())
                    .json(&req_body)
                    .send()
                    .await {
                        Ok(r) => r,
                        Err(e) => {
                            let _ = tx.send(Err(BackendError::ConnectionFailed(e.to_string()))).await;
                            return;
                        }
                    };

                if !resp.status().is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    let _ = tx.send(Err(BackendError::RequestFailed(body))).await;
                    return;
                }

                // Ollama streams newline-delimited JSON
                use futures_util::StreamExt;
                let mut stream = resp.bytes_stream();
                let mut byte_buffer = Vec::new();

                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(bytes) => {
                            byte_buffer.extend_from_slice(&bytes);
                            // Process complete JSON lines
                            while let Some(newline_pos) = byte_buffer.iter().position(|&b| b == b'\n') {
                                let line_bytes: Vec<u8> = byte_buffer.drain(..=newline_pos).collect();
                                let line = String::from_utf8_lossy(&line_bytes).trim().to_string();

                                if line.is_empty() { continue; }

                                if let Ok(ollama_chunk) = serde_json::from_str::<OllamaChatResponse>(&line) {
                                    let content = ollama_chunk.message
                                        .as_ref()
                                        .map(|m| m.content.clone())
                                        .unwrap_or_default();

                                    let done = ollama_chunk.done.unwrap_or(false);

                                    let tok_per_sec = if done {
                                        if let (Some(count), Some(dur)) = (ollama_chunk.eval_count, ollama_chunk.eval_duration) {
                                            if dur > 0 { (count as f64 / dur as f64) * 1_000_000_000.0 } else { 0.0 }
                                        } else { 0.0 }
                                    } else { 0.0 };

                                    if tx.send(Ok(StreamChunk {
                                        content,
                                        done,
                                        tok_per_sec,
                                        eval_tokens: ollama_chunk.eval_count.unwrap_or(0),
                                    })).await.is_err() {
                                        return; // Client disconnected
                                    }

                                    if done { return; }
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(BackendError::ConnectionFailed(e.to_string()))).await;
                            return;
                        }
                    }
                }
            }
            BackendType::OpenAICompatible => {
                let req_body = OpenAIChatRequest {
                    model: resolved_model,
                    messages: chat_messages,
                    stream: true,
                    temperature: Some(temperature),
                    top_p: None,
                    max_tokens,
                };

                let mut request = self.client.post(self.config.chat_url()).json(&req_body);
                if let Some(ref key) = self.config.api_key {
                    request = request.header("Authorization", format!("Bearer {}", key));
                }

                let resp = match request.send().await {
                    Ok(r) => r,
                    Err(e) => {
                        let _ = tx.send(Err(BackendError::ConnectionFailed(e.to_string()))).await;
                        return;
                    }
                };

                if !resp.status().is_success() {
                    let body = resp.text().await.unwrap_or_default();
                    let _ = tx.send(Err(BackendError::RequestFailed(body))).await;
                    return;
                }

                use futures_util::StreamExt;
                let mut stream = resp.bytes_stream();
                let mut byte_buffer = Vec::new();

                while let Some(chunk_result) = stream.next().await {
                    match chunk_result {
                        Ok(bytes) => {
                            byte_buffer.extend_from_slice(&bytes);
                            while let Some(newline_pos) = byte_buffer.iter().position(|&b| b == b'\n') {
                                let line_bytes: Vec<u8> = byte_buffer.drain(..=newline_pos).collect();
                                let line = String::from_utf8_lossy(&line_bytes).trim().to_string();

                                if !line.starts_with("data: ") { continue; }
                                let json_str = &line[6..];
                                if json_str == "[DONE]" {
                                    let _ = tx.send(Ok(StreamChunk {
                                        content: String::new(),
                                        done: true,
                                        tok_per_sec: 0.0,
                                        eval_tokens: 0,
                                    })).await;
                                    return;
                                }

                                if let Ok(v) = serde_json::from_str::<serde_json::Value>(json_str) {
                                    let content = v["choices"][0]["delta"]["content"]
                                        .as_str()
                                        .unwrap_or("")
                                        .to_string();

                                    if tx.send(Ok(StreamChunk {
                                        content,
                                        done: false,
                                        tok_per_sec: 0.0,
                                        eval_tokens: 0,
                                    })).await.is_err() {
                                        return;
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Err(BackendError::ConnectionFailed(e.to_string()))).await;
                            return;
                        }
                    }
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct StreamChunk {
    pub content: String,
    pub done: bool,
    pub tok_per_sec: f64,
    pub eval_tokens: u64,
}

#[derive(Debug, Clone, Default)]
pub struct BackendMetrics {
    pub eval_tokens: u64,
    pub eval_duration_ns: u64,
    pub prompt_eval_tokens: u64,
    pub prompt_eval_duration_ns: u64,
    pub tok_per_sec: f64,
    pub ttft_ms: f64,
}

#[derive(Debug, Clone)]
pub enum BackendError {
    ConnectionFailed(String),
    RequestFailed(String),
    ParseError(String),
    NoBackendAvailable,
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ConnectionFailed(e) => write!(f, "Backend connection failed: {}", e),
            Self::RequestFailed(e) => write!(f, "Backend request failed: {}", e),
            Self::ParseError(e) => write!(f, "Response parse error: {}", e),
            Self::NoBackendAvailable => write!(f, "No LLM backend is running. Start Ollama (`ollama serve`) or any OpenAI-compatible server."),
        }
    }
}
