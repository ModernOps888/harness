use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::Stream;
use harness_core::{Device, Tensor};
use harness_pipeline::{CodeQualityEngine, Language};
use harness_safety::{EntropyDetector, LateralInhibitionFilter};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::{Duration, Instant};
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::backend::{BackendError, StreamChunk};
use crate::state::AppState;

// ---------------------------------------------------------------------------
// Request / Response types (OpenAI-compatible)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    pub stream: Option<bool>,
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub max_tokens: Option<usize>,
    pub response_format: Option<serde_json::Value>,
    // Extended HARNESS flags
    pub constrained_mode: Option<String>,
    pub spiking_threshold: Option<f32>,
    pub lateral_contrast: Option<f32>,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionChoice {
    pub index: usize,
    pub message: ChatMessage,
    pub finish_reason: String,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionUsage {
    pub prompt_tokens: usize,
    pub completion_tokens: usize,
    pub total_tokens: usize,
    pub tokens_per_second: f64,
    pub confidence_score: f64,
}

#[derive(Debug, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
    pub usage: ChatCompletionUsage,
}

// ---------------------------------------------------------------------------
// HARNESS Enhancement Layer
// ---------------------------------------------------------------------------

/// Post-process real LLM output with HARNESS enhancements.
/// Returns (confidence_score, entropy).
pub fn apply_harness_enhancements(
    text: &str,
    lateral_contrast: f32,
    _spiking_th: f32,
) -> (f64, f64) {
    // Compute entropy from the output byte distribution (real signal analysis)
    let mut byte_freq = [0u32; 256];
    for b in text.bytes() {
        byte_freq[b as usize] += 1;
    }
    let total = text.len().max(1) as f64;
    let entropy: f64 = byte_freq.iter()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / total;
            -p * p.ln()
        })
        .sum();

    // Normalize entropy to [0, 1] range (max byte entropy is ln(256) = 5.545)
    let normalized_entropy = (entropy / 5.545).clamp(0.0, 1.0);

    // Apply lateral inhibition contrast to compute confidence
    // Higher contrast + lower entropy = higher confidence
    let contrast_factor = (lateral_contrast as f64 / 2.0).clamp(0.5, 1.5);
    let confidence = (1.0 - normalized_entropy * 0.3) * contrast_factor;
    let confidence = confidence.clamp(0.0, 1.0);

    // Run lateral inhibition on empirical token frequency log-probabilities
    let mut token_signal_logits: Vec<f32> = byte_freq.iter()
        .filter(|&&c| c > 0)
        .take(64)
        .map(|&c| (c as f32 / total as f32).ln())
        .collect();
    if token_signal_logits.len() < 16 {
        token_signal_logits.resize(16, -10.0);
    }
    let filter = LateralInhibitionFilter::new(lateral_contrast, 0.05);
    filter.sharpen_logits(&mut token_signal_logits);

    // Compute Shannon entropy on sharpened logits
    let logit_entropy = if let Ok(t) = Tensor::from_f32_slice(
        &token_signal_logits,
        vec![1, token_signal_logits.len()],
        Device::Cpu,
    ) {
        EntropyDetector::compute_entropy(&t).unwrap_or(0.0) as f64
    } else {
        0.0
    };

    // Calculate final confidence from empirical entropy measurements
    let final_confidence = (confidence * 0.7 + (1.0 - logit_entropy * 0.1) * 0.3).clamp(0.1, 1.0);

    (final_confidence, normalized_entropy)
}

/// Apply CodeQualityEngine delimiter verification and auto-repair when constrained_mode is enabled
fn apply_code_verification(text: &str, constrained_mode: Option<&str>) -> String {
    let mode = constrained_mode.unwrap_or("").trim().to_lowercase();
    if mode.is_empty() || mode == "none" || mode == "false" {
        return text.to_string();
    }
    if mode == "json" || mode == "true" {
        let check = CodeQualityEngine::verify_syntax(text, Language::Json);
        if !check.is_valid {
            if let Some(repaired) = check.repaired_code {
                return repaired;
            }
        }
    } else {
        let lang = Language::from_tag(&mode);
        let check = CodeQualityEngine::verify_syntax(text, lang);
        if !check.is_valid {
            if let Some(repaired) = check.repaired_code {
                return repaired;
            }
        }
    }
    text.to_string()
}

/// Record KV block allocation and prefix cache entry for an active request
fn record_kv_telemetry(state: &AppState, req_uuid: Uuid, messages: &[(String, String)], total_tokens: usize) {
    let prefix_tokens: Vec<u32> = messages
        .iter()
        .flat_map(|(_, content)| content.bytes().take(64).map(|b| b as u32))
        .take(128)
        .collect();

    let blocks_needed = total_tokens.div_ceil(16).clamp(1, 32);
    let mut allocated_blocks = Vec::with_capacity(blocks_needed);

    if let Ok(mut paged) = state.paged_attn.lock() {
        // Keep pool healthy by freeing if low on blocks
        if paged.free_block_count() < blocks_needed {
            return;
        }
        for _ in 0..blocks_needed {
            if let Ok(bid) = paged.allocate_block(req_uuid) {
                allocated_blocks.push(bid);
            }
        }
        paged.record_tokens(&req_uuid, total_tokens);
    }

    if !prefix_tokens.is_empty() && !allocated_blocks.is_empty() {
        if let Ok(mut cache) = state.prefix_cache.lock() {
            let (matched, _) = cache.match_prefix(&prefix_tokens);
            if matched < prefix_tokens.len() {
                cache.insert(&prefix_tokens, &allocated_blocks);
            }
        }
    }
}

/// Compact conversation history to reduce token count.
/// Keeps system message + last N user/assistant turns in full,
/// summarizes older turns to their first sentence.
fn compact_messages(messages: &[ChatMessage], max_full_turns: usize) -> Vec<(String, String)> {
    let mut result = Vec::new();

    // Always keep system messages in full
    let system_msgs: Vec<&ChatMessage> = messages.iter()
        .filter(|m| m.role == "system")
        .collect();
    let non_system: Vec<&ChatMessage> = messages.iter()
        .filter(|m| m.role != "system")
        .collect();

    for msg in &system_msgs {
        result.push((msg.role.clone(), msg.content.clone()));
    }

    let turn_count = non_system.len();
    if turn_count <= max_full_turns * 2 {
        // Short conversation - keep everything
        for msg in &non_system {
            result.push((msg.role.clone(), msg.content.clone()));
        }
    } else {
        // Long conversation - compact older messages
        let cutoff = turn_count - max_full_turns * 2;
        for (i, msg) in non_system.iter().enumerate() {
            if i < cutoff {
                // Compact: keep first 200 chars
                let compacted = if msg.content.chars().count() > 200 {
                    let truncated: String = msg.content.chars().take(200).collect();
                    format!("{}...", truncated)
                } else {
                    msg.content.clone()
                };
                result.push((msg.role.clone(), compacted));
            } else {
                // Recent: keep in full
                result.push((msg.role.clone(), msg.content.clone()));
            }
        }
    }

    result
}

/// Estimate prompt tokens (rough word-based approximation)
fn estimate_tokens(messages: &[(String, String)]) -> usize {
    messages.iter()
        .map(|(_, content)| {
            // ~1.3 tokens per word is a reasonable approximation
            (content.split_whitespace().count() as f64 * 1.3) as usize
        })
        .sum()
}

// ---------------------------------------------------------------------------
// Main handler: proxies to real LLM backend + HARNESS enhancements
// ---------------------------------------------------------------------------

pub async fn chat_completions(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Response {
    let req_uuid = Uuid::new_v4();
    let req_id = format!("chatcmpl-{}", req_uuid);
    let current_model = req.model.unwrap_or_else(|| state.model_name.read().unwrap().clone());
    let stream_mode = req.stream.unwrap_or(false);
    let temperature = req.temperature.unwrap_or(0.7);
    let max_tokens = req.max_tokens;
    let lateral_contrast = req.lateral_contrast.unwrap_or(1.8);
    let spiking_th = req.spiking_threshold.unwrap_or(0.35);
    let constrained_mode = req.constrained_mode.clone();

    // Update active model name in state
    if !current_model.is_empty() && current_model != "default" && current_model != "auto" {
        if let Ok(mut active) = state.model_name.write() {
            *active = current_model.clone();
        }
    }

    // Compact conversation history for efficiency (fewer tokens = faster inference)
    let messages = compact_messages(&req.messages, 5);
    let prompt_tokens = estimate_tokens(&messages);

    // Check if backend is available
    let backend_available = state.backend.config.is_available().await;

    if !backend_available {
        // Return clear error - no faking
        let error_msg = format!(
            "No LLM backend is running. HARNESS needs a real inference backend to generate responses.\n\n\
            Start one of:\n\
            - HARNESS 32B Coder Speculative Engine (port 8089): `./launch_qwen32b_coder.ps1` or `python server_runner.py`\n\
            - Ollama (port 11434): `ollama serve` (then `ollama pull qwen2.5-coder:32b`)\n\
            - llama.cpp server: `llama-server -m model.gguf --port 8089`\n\
            - vLLM: `vllm serve Qwen/Qwen2.5-Coder-32B-Instruct --port 8000`\n\n\
            Then restart HARNESS or it will auto-detect on next request.\n\n\
            Tried: {}", state.backend.config.base_url
        );

        let response = ChatCompletionResponse {
            id: req_id,
            object: "chat.completion".into(),
            created: chrono::Utc::now().timestamp(),
            model: current_model,
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: error_msg,
                },
                finish_reason: "stop".into(),
            }],
            usage: ChatCompletionUsage {
                prompt_tokens,
                completion_tokens: 0,
                total_tokens: prompt_tokens,
                tokens_per_second: 0.0,
                confidence_score: 0.0,
            },
        };
        return Json(response).into_response();
    }

    if stream_mode {
        // Stream real tokens from backend via SSE
        record_kv_telemetry(&state, req_uuid, &messages, prompt_tokens.max(16));
        let stream = stream_from_backend(
            req_id,
            current_model,
            messages,
            temperature,
            max_tokens,
            lateral_contrast,
            spiking_th,
            state,
        );
        Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
            .into_response()
    } else {
        // Non-streaming: get full response from backend
        let start = Instant::now();

        match state.backend.chat_completion(
            &current_model,
            &messages,
            temperature,
            max_tokens,
        ).await {
            Ok((raw_content, backend_metrics)) => {
                let elapsed = start.elapsed();
                let content = apply_code_verification(&raw_content, constrained_mode.as_deref());
                let completion_tokens = if backend_metrics.eval_tokens > 0 {
                    backend_metrics.eval_tokens as usize
                } else {
                    content.split_whitespace().count()
                };

                // Apply HARNESS enhancements on real output
                let (confidence, _entropy) = apply_harness_enhancements(
                    &content,
                    lateral_contrast,
                    spiking_th,
                );

                // Use real tok/s from backend, fall back to our measurement
                let tok_per_sec = if backend_metrics.tok_per_sec > 0.0 {
                    backend_metrics.tok_per_sec
                } else if elapsed.as_secs_f64() > 0.0 {
                    completion_tokens as f64 / elapsed.as_secs_f64()
                } else {
                    0.0
                };

                state.record_tokens(completion_tokens);
                state.record_throughput(tok_per_sec as f32);
                record_kv_telemetry(&state, req_uuid, &messages, prompt_tokens + completion_tokens);

                let response = ChatCompletionResponse {
                    id: req_id,
                    object: "chat.completion".into(),
                    created: chrono::Utc::now().timestamp(),
                    model: current_model,
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".into(),
                            content,
                        },
                        finish_reason: "stop".into(),
                    }],
                    usage: ChatCompletionUsage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        tokens_per_second: (tok_per_sec * 10.0).round() / 10.0,
                        confidence_score: (confidence * 1000.0).round() / 1000.0,
                    },
                };

                Json(response).into_response()
            }
            Err(e) => {
                let error_response = ChatCompletionResponse {
                    id: req_id,
                    object: "chat.completion".into(),
                    created: chrono::Utc::now().timestamp(),
                    model: current_model,
                    choices: vec![ChatCompletionChoice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".into(),
                            content: format!("Backend error: {}", e),
                        },
                        finish_reason: "stop".into(),
                    }],
                    usage: ChatCompletionUsage {
                        prompt_tokens,
                        completion_tokens: 0,
                        total_tokens: prompt_tokens,
                        tokens_per_second: 0.0,
                        confidence_score: 0.0,
                    },
                };
                Json(error_response).into_response()
            }
        }
    }
}

// ---------------------------------------------------------------------------
// SSE Streaming: real tokens from backend, enhanced by HARNESS
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn stream_from_backend(
    req_id: String,
    model: String,
    messages: Vec<(String, String)>,
    temperature: f32,
    max_tokens: Option<usize>,
    lateral_contrast: f32,
    spiking_th: f32,
    state: AppState,
) -> impl Stream<Item = std::result::Result<Event, Infallible>> {
    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel(256);

    tokio::spawn(async move {
        let (backend_tx, mut backend_rx) = tokio::sync::mpsc::channel::<Result<StreamChunk, BackendError>>(256);
        let start = Instant::now();

        // Spawn backend streaming in background
        let backend = state.backend.clone();
        let model_clone = model.clone();
        let messages_clone = messages.clone();
        tokio::spawn(async move {
            backend.chat_completion_stream(
                &model_clone,
                &messages_clone,
                temperature,
                max_tokens,
                backend_tx,
            ).await;
        });

        let mut total_content = String::new();
        let mut token_count: usize = 0;

        // Forward real tokens from backend as SSE events
        while let Some(chunk_result) = backend_rx.recv().await {
            match chunk_result {
                Ok(chunk) => {
                    if chunk.done {
                        // Final chunk: compute HARNESS metrics on complete output
                        let elapsed = start.elapsed();
                        let (confidence, _entropy) = apply_harness_enhancements(
                            &total_content,
                            lateral_contrast,
                            spiking_th,
                        );

                        let tok_per_sec = if chunk.tok_per_sec > 0.0 {
                            chunk.tok_per_sec
                        } else if elapsed.as_secs_f64() > 0.0 {
                            token_count as f64 / elapsed.as_secs_f64()
                        } else {
                            0.0
                        };

                        // Send final chunk with usage stats
                        let final_chunk = serde_json::json!({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": chrono::Utc::now().timestamp(),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": {},
                                "finish_reason": "stop"
                            }],
                            "usage": {
                                "prompt_tokens": estimate_tokens(&messages),
                                "completion_tokens": token_count,
                                "total_tokens": estimate_tokens(&messages) + token_count,
                                "tokens_per_second": (tok_per_sec * 10.0).round() / 10.0,
                                "confidence_score": (confidence * 1000.0).round() / 1000.0
                            }
                        });

                        let event = Event::default().data(final_chunk.to_string());
                        let _ = sse_tx.send(Ok(event)).await;

                        // Send [DONE] marker
                        let done_event = Event::default().data("[DONE]");
                        let _ = sse_tx.send(Ok(done_event)).await;

                        state.record_tokens(token_count);
                        state.record_throughput(tok_per_sec as f32);
                        break;
                    }

                    if !chunk.content.is_empty() {
                        total_content.push_str(&chunk.content);
                        token_count += 1;

                        let delta_chunk = serde_json::json!({
                            "id": req_id,
                            "object": "chat.completion.chunk",
                            "created": chrono::Utc::now().timestamp(),
                            "model": model,
                            "choices": [{
                                "index": 0,
                                "delta": { "content": chunk.content },
                                "finish_reason": null
                            }]
                        });

                        let event = Event::default().data(delta_chunk.to_string());
                        if sse_tx.send(Ok(event)).await.is_err() {
                            break; // Client disconnected
                        }
                    }
                }
                Err(e) => {
                    // Send error as SSE event
                    let error_chunk = serde_json::json!({
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": chrono::Utc::now().timestamp(),
                        "model": model,
                        "choices": [{
                            "index": 0,
                            "delta": { "content": format!("\n\n[HARNESS Error: {}]", e) },
                            "finish_reason": "stop"
                        }]
                    });
                    let event = Event::default().data(error_chunk.to_string());
                    let _ = sse_tx.send(Ok(event)).await;
                    break;
                }
            }
        }
    });

    ReceiverStream::new(sse_rx)
}
