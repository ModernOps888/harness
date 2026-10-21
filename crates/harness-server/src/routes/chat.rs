use axum::body::Body;
use axum::extract::State;
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::Json;
use futures_util::{Stream, StreamExt};
use harness_core::{Device, Tensor};
use harness_pipeline::{CodeQualityEngine, Language};
use harness_safety::{EntropyDetector, LateralInhibitionFilter};
use serde_json::{json, Value};
use std::convert::Infallible;
use std::time::Duration;
use tokio_stream::wrappers::ReceiverStream;
use uuid::Uuid;

use crate::backend::BackendError;
use crate::state::AppState;

// ---------------------------------------------------------------------------
// HARNESS Enhancement Layer
// ---------------------------------------------------------------------------

/// Post-process real LLM output with HARNESS enhancements.
/// Returns (confidence_score, entropy).
///
/// NOTE: `confidence_score` is a *heuristic* derived from the byte-level Shannon entropy of the
/// generated text sharpened by the lateral-inhibition filter. It is NOT a measured factuality
/// or correctness probability.
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
pub fn apply_code_verification(text: &str, constrained_mode: Option<&str>) -> String {
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
fn record_kv_telemetry(state: &AppState, req_uuid: Uuid, prefix_source: &str, total_tokens: usize) {
    let prefix_tokens: Vec<u32> = prefix_source.bytes().take(128).map(|b| b as u32).collect();

    let blocks_needed = total_tokens.div_ceil(16).clamp(1, 32);
    let mut allocated_blocks = Vec::with_capacity(blocks_needed);

    if let Ok(mut paged) = state.paged_attn.lock() {
        // Keep pool healthy by skipping when low on blocks
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

/// Plain text of a message `content` value, which per the OpenAI spec may be a string,
/// `null` (assistant tool-call turns), or an array of typed content parts.
fn content_text(content: &Value) -> String {
    match content {
        Value::String(s) => s.clone(),
        Value::Array(parts) => parts
            .iter()
            .filter_map(|p| p.get("text").and_then(|t| t.as_str()))
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// OpenAI-format error body so IDE clients show a real error instead of a fake assistant turn.
pub fn openai_error(status: StatusCode, message: impl Into<String>, kind: &str) -> Response {
    let body = json!({
        "error": {
            "message": message.into(),
            "type": kind,
            "param": Value::Null,
            "code": status.as_u16(),
        }
    });
    (status, Json(body)).into_response()
}

fn backend_error_response(base_url: &str, e: &BackendError) -> Response {
    openai_error(
        StatusCode::SERVICE_UNAVAILABLE,
        format!(
            "{} (backend: {}). Start the HARNESS engine with `./launch_qwen32b_coder.ps1` \
             (llama-server on :8089) or Ollama, then retry.",
            e, base_url
        ),
        "backend_unavailable",
    )
}

/// Extension keys understood by HARNESS only; stripped before the body reaches the backend.
struct HarnessFlags {
    constrained_mode: Option<String>,
    lateral_contrast: f32,
    spiking_threshold: f32,
}

fn take_harness_flags(body: &mut Value) -> HarnessFlags {
    let obj = body.as_object_mut();
    let (mut cm, mut lc, mut st) = (None, 1.8f32, 0.35f32);
    if let Some(obj) = obj {
        if let Some(v) = obj.remove("constrained_mode") {
            cm = v.as_str().map(|s| s.to_string());
        }
        if let Some(v) = obj.remove("lateral_contrast") {
            lc = v.as_f64().map(|f| f as f32).unwrap_or(lc);
        }
        if let Some(v) = obj.remove("spiking_threshold") {
            st = v.as_f64().map(|f| f as f32).unwrap_or(st);
        }
    }
    HarnessFlags { constrained_mode: cm, lateral_contrast: lc, spiking_threshold: st }
}

// ---------------------------------------------------------------------------
// Main handler: lossless OpenAI proxy (tools, tool_calls, multimodal parts, stop, ...)
// ---------------------------------------------------------------------------

/// `POST /v1/chat/completions`
///
/// The request body is forwarded **verbatim** to the inference backend (only `model` is resolved
/// to the id the backend actually serves and HARNESS-only extension keys are removed). That keeps
/// every OpenAI feature IDE clients rely on intact: `tools`, `tool_choice`, `tool_calls`,
/// `role: "tool"` messages, `content: null`, content-part arrays, `stop`, `stream_options`, ...
pub async fn chat_completions(
    State(state): State<AppState>,
    Json(mut body): Json<Value>,
) -> Response {
    let req_uuid = Uuid::new_v4();
    let flags = take_harness_flags(&mut body);

    if !body.get("messages").map(|m| m.is_array()).unwrap_or(false) {
        return openai_error(StatusCode::BAD_REQUEST, "`messages` must be an array", "invalid_request_error");
    }

    let requested_model = body
        .get("model")
        .and_then(|m| m.as_str())
        .map(|s| s.to_string())
        .unwrap_or_else(|| state.model_name.read().unwrap().clone());
    let stream_mode = body.get("stream").and_then(|s| s.as_bool()).unwrap_or(false);

    let resolved = state.backend.resolve_model_smart(&requested_model).await;
    body["model"] = Value::String(resolved.clone());
    if !requested_model.is_empty() && requested_model != "default" && requested_model != "auto" {
        if let Ok(mut active) = state.model_name.write() {
            *active = requested_model.clone();
        }
    }

    // KV/prefix bookkeeping (telemetry only) from the real prompt text.
    let prompt_text: String = body["messages"]
        .as_array()
        .map(|msgs| {
            msgs.iter()
                .map(|m| content_text(&m["content"]))
                .collect::<Vec<_>>()
                .join("\n")
        })
        .unwrap_or_default();
    record_kv_telemetry(&state, req_uuid, &prompt_text, (prompt_text.split_whitespace().count() as f64 * 1.3) as usize + 16);

    let base_url = state.backend.config.base_url.clone();
    let backend = state.backend.clone();
    let body_for_task = body.clone();
    let mut pending = tokio::spawn(async move { backend.forward_json("v1/chat/completions", &body_for_task).await });

    // llama-server sends response headers only after the prompt has been prefilled (minutes for a
    // 10k-token agent prompt). Fast outcomes (4xx overflow, connection refused) keep their proper
    // HTTP status; if headers are not back within 2 s, answer with an SSE stream right away so
    // clients with header timeouts (Node/undici: 300 s) survive, and KeepAlive comments flow.
    let early = tokio::time::timeout(Duration::from_secs(2), &mut pending).await;
    let upstream = match early {
        Ok(Ok(Ok(r))) => Some(r),
        Ok(Ok(Err(e))) => return backend_error_response(&base_url, &e),
        Ok(Err(e)) => return openai_error(StatusCode::INTERNAL_SERVER_ERROR, format!("proxy task failed: {e}"), "internal_error"),
        Err(_elapsed) => None,
    };

    if let Some(up) = &upstream {
        let status = StatusCode::from_u16(up.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
        if !status.is_success() {
            let text = upstream.unwrap().text().await.unwrap_or_default();
            let mut resp = Response::new(Body::from(text));
            *resp.status_mut() = status;
            resp.headers_mut()
                .insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
            return resp;
        }
    }

    if stream_mode {
        let source = match upstream {
            Some(r) => UpstreamSource::Ready(r),
            None => UpstreamSource::Pending(pending),
        };
        let stream = stream_passthrough(source, flags, state);
        return Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(10)))
            .into_response();
    }

    // Non-streaming: wait for the (possibly slow) upstream headers + body.
    let upstream = match upstream {
        Some(r) => r,
        None => match pending.await {
            Ok(Ok(r)) => r,
            Ok(Err(e)) => return backend_error_response(&base_url, &e),
            Err(e) => return openai_error(StatusCode::INTERNAL_SERVER_ERROR, format!("proxy task failed: {e}"), "internal_error"),
        },
    };
    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    if !status.is_success() {
        let text = upstream.text().await.unwrap_or_default();
        let mut resp = Response::new(Body::from(text));
        *resp.status_mut() = status;
        resp.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        return resp;
    }

    let mut json_resp: Value = match upstream.json().await {
        Ok(v) => v,
        Err(e) => {
            return openai_error(StatusCode::BAD_GATEWAY, format!("Backend returned invalid JSON: {e}"), "bad_gateway")
        }
    };

    let has_tool_calls = json_resp["choices"][0]["message"]["tool_calls"]
        .as_array()
        .map(|a| !a.is_empty())
        .unwrap_or(false);
    let raw_content = json_resp["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string();

    // Only rewrite plain-text answers; never touch tool-call turns.
    if !has_tool_calls && !raw_content.is_empty() {
        let verified = apply_code_verification(&raw_content, flags.constrained_mode.as_deref());
        if verified != raw_content {
            json_resp["choices"][0]["message"]["content"] = Value::String(verified);
        }
    }
    let final_text = json_resp["choices"][0]["message"]["content"].as_str().unwrap_or("").to_string();

    let tok_per_sec = json_resp["timings"]["predicted_per_second"].as_f64().unwrap_or(0.0);
    let completion_tokens = json_resp["usage"]["completion_tokens"].as_u64().unwrap_or(0) as usize;
    if !json_resp["usage"].is_object() {
        json_resp["usage"] = json!({});
    }
    json_resp["usage"]["tokens_per_second"] = json!((tok_per_sec * 10.0).round() / 10.0);
    if !final_text.is_empty() {
        let (confidence, _) = apply_harness_enhancements(&final_text, flags.lateral_contrast, flags.spiking_threshold);
        json_resp["usage"]["confidence_score"] = json!((confidence * 1000.0).round() / 1000.0);
    }

    state.record_tokens(completion_tokens);
    state.record_throughput(tok_per_sec as f32);

    Json(json_resp).into_response()
}

// ---------------------------------------------------------------------------
// SSE streaming: forward every upstream `data:` event unchanged (tool_call deltas included);
// attach HARNESS metrics to the chunk that carries `finish_reason`.
// ---------------------------------------------------------------------------

/// Upstream connection that is either established or still waiting for response headers.
enum UpstreamSource {
    Ready(reqwest::Response),
    Pending(tokio::task::JoinHandle<Result<reqwest::Response, BackendError>>),
}

fn stream_passthrough(
    source: UpstreamSource,
    flags: HarnessFlags,
    state: AppState,
) -> impl Stream<Item = std::result::Result<Event, Infallible>> {
    let (sse_tx, sse_rx) = tokio::sync::mpsc::channel::<std::result::Result<Event, Infallible>>(256);

    tokio::spawn(async move {
        let upstream = match source {
            UpstreamSource::Ready(r) => r,
            UpstreamSource::Pending(mut handle) => {
                // Still prefilling: wait for headers, but drop the upstream request if the client leaves.
                let joined = tokio::select! {
                    _ = sse_tx.closed() => { handle.abort(); return; }
                    j = &mut handle => j,
                };
                let resp = match joined {
                    Ok(Ok(r)) => r,
                    Ok(Err(e)) => {
                        let err = json!({"error": {"message": e.to_string(), "type": "backend_unavailable"}});
                        let _ = sse_tx.send(Ok(Event::default().data(err.to_string()))).await;
                        return;
                    }
                    Err(e) => {
                        let err = json!({"error": {"message": format!("proxy task failed: {e}"), "type": "internal_error"}});
                        let _ = sse_tx.send(Ok(Event::default().data(err.to_string()))).await;
                        return;
                    }
                };
                if !resp.status().is_success() {
                    let status = resp.status().as_u16();
                    let text = resp.text().await.unwrap_or_default();
                    let msg = serde_json::from_str::<Value>(&text)
                        .ok()
                        .and_then(|v| v["error"]["message"].as_str().map(|s| s.to_string()))
                        .unwrap_or(text);
                    let err = json!({"error": {"message": format!("Backend HTTP {status}: {msg}"), "type": "backend_error", "code": status}});
                    let _ = sse_tx.send(Ok(Event::default().data(err.to_string()))).await;
                    return;
                }
                resp
            }
        };
        let mut bytes = upstream.bytes_stream();
        let mut buf: Vec<u8> = Vec::new();
        let mut content = String::new();
        let mut delta_events: usize = 0;
        let mut tok_per_sec = 0.0f64;
        let mut completion_tokens: usize = 0;

        loop {
            // llama-server emits nothing while it prefills a long prompt (minutes at 60-80 tok/s).
            // Race the upstream against client disconnect so a cancelled request is dropped
            // (dropping `bytes` closes the upstream connection, which aborts the generation and
            // frees the single slot) instead of being noticed only on the next token.
            let next = tokio::select! {
                _ = sse_tx.closed() => return,
                n = bytes.next() => n,
            };
            let Some(chunk) = next else { break };
            let chunk = match chunk {
                Ok(b) => b,
                Err(e) => {
                    let err = json!({"error": {"message": format!("Backend stream error: {e}"), "type": "backend_stream_error"}});
                    let _ = sse_tx.send(Ok(Event::default().data(err.to_string()))).await;
                    break;
                }
            };
            buf.extend_from_slice(&chunk);

            while let Some(pos) = buf.iter().position(|&b| b == b'\n') {
                let line_bytes: Vec<u8> = buf.drain(..=pos).collect();
                let line = String::from_utf8_lossy(&line_bytes).trim().to_string();
                let Some(payload) = line.strip_prefix("data:") else { continue };
                let payload = payload.trim_start();

                if payload == "[DONE]" {
                    let _ = sse_tx.send(Ok(Event::default().data("[DONE]"))).await;
                    state.record_tokens(if completion_tokens > 0 { completion_tokens } else { delta_events });
                    state.record_throughput(tok_per_sec as f32);
                    return;
                }

                let out = match serde_json::from_str::<Value>(payload) {
                    Ok(mut v) => {
                        if let Some(t) = v["choices"][0]["delta"]["content"].as_str() {
                            if !t.is_empty() {
                                content.push_str(t);
                                delta_events += 1;
                            }
                        }
                        if let Some(t) = v["timings"]["predicted_per_second"].as_f64() {
                            tok_per_sec = t;
                        }
                        if let Some(n) = v["usage"]["completion_tokens"].as_u64() {
                            completion_tokens = n as usize;
                        }
                        // Attach HARNESS metrics (top-level `harness` key, never inside `usage`, so strict
                        // OpenAI SDK parsers keep validating) to the finish chunk and to the trailing
                        // usage chunk (empty `choices`) where llama-server reports its `timings`.
                        let choices_empty = v["choices"].as_array().map(|a| a.is_empty()).unwrap_or(false);
                        let finished = !v["choices"][0]["finish_reason"].is_null();
                        if finished || (choices_empty && v["usage"].is_object()) {
                            let mut h = json!({});
                            if tok_per_sec > 0.0 {
                                h["tokens_per_second"] = json!((tok_per_sec * 10.0).round() / 10.0);
                            }
                            if !content.is_empty() {
                                let (confidence, _) = apply_harness_enhancements(
                                    &content,
                                    flags.lateral_contrast,
                                    flags.spiking_threshold,
                                );
                                h["confidence_score"] = json!((confidence * 1000.0).round() / 1000.0);
                            }
                            v["harness"] = h;
                        }
                        v.to_string()
                    }
                    Err(_) => payload.to_string(),
                };
                if sse_tx.send(Ok(Event::default().data(out))).await.is_err() {
                    return; // client disconnected
                }
            }
        }
        // Reaching here means the upstream closed WITHOUT sending [DONE] (the [DONE] path returns above):
        // surface it instead of letting clients treat a truncated generation as a clean finish.
        let err = json!({"error": {"message": "Backend stream ended unexpectedly (no [DONE])", "type": "backend_stream_truncated"}});
        let _ = sse_tx.send(Ok(Event::default().data(err.to_string()))).await;
    });

    ReceiverStream::new(sse_rx)
}

// ---------------------------------------------------------------------------
// Other OpenAI-compatible endpoints: raw passthrough (FIM / code completion, embeddings)
// ---------------------------------------------------------------------------

async fn passthrough(state: AppState, upstream_path: &str, mut body: Value) -> Response {
    let _ = take_harness_flags(&mut body);
    if let Some(requested) = body.get("model").and_then(|m| m.as_str()).map(|s| s.to_string()) {
        body["model"] = Value::String(state.backend.resolve_model_smart(&requested).await);
    }
    let base_url = state.backend.config.base_url.clone();
    let upstream = match state.backend.forward_json(upstream_path, &body).await {
        Ok(r) => r,
        Err(e) => return backend_error_response(&base_url, &e),
    };
    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream
        .headers()
        .get(header::CONTENT_TYPE)
        .cloned()
        .unwrap_or(HeaderValue::from_static("application/json"));
    let mut resp = Response::new(Body::from_stream(upstream.bytes_stream()));
    *resp.status_mut() = status;
    resp.headers_mut().insert(header::CONTENT_TYPE, content_type);
    resp
}

/// `POST /v1/completions` — raw text / fill-in-the-middle completion (editor autocomplete).
pub async fn completions(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(state, "v1/completions", body).await
}

/// `POST /v1/embeddings`
pub async fn embeddings(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(state, "v1/embeddings", body).await
}

/// `POST /tokenize` ? exact token counts from the loaded model's tokenizer (llama-server).
pub async fn tokenize(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(state, "tokenize", body).await
}

/// `POST /apply-template` ? render chat messages with the model's own chat template (token budgeting).
pub async fn apply_template(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(state, "apply-template", body).await
}

/// `POST /infill` — llama.cpp native code infilling (prefix/suffix) used by FIM-capable editors.
pub async fn infill(State(state): State<AppState>, Json(body): Json<Value>) -> Response {
    passthrough(state, "infill", body).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn content_text_handles_string_null_and_parts() {
        assert_eq!(content_text(&json!("hi")), "hi");
        assert_eq!(content_text(&Value::Null), "");
        assert_eq!(
            content_text(&json!([{"type":"text","text":"a"},{"type":"image_url","image_url":{"url":"x"}},{"type":"text","text":"b"}])),
            "a\nb"
        );
    }

    #[test]
    fn harness_flags_are_stripped_and_rest_untouched() {
        let mut body = json!({
            "model": "m",
            "messages": [{"role":"assistant","content":null,"tool_calls":[{"id":"1","type":"function","function":{"name":"f","arguments":"{}"}}]}],
            "tools": [{"type":"function","function":{"name":"f","parameters":{"type":"object"}}}],
            "constrained_mode": "rust",
            "lateral_contrast": 2.5,
            "spiking_threshold": 0.5
        });
        let flags = take_harness_flags(&mut body);
        assert_eq!(flags.constrained_mode.as_deref(), Some("rust"));
        assert!((flags.lateral_contrast - 2.5).abs() < 1e-6);
        assert!(body.get("constrained_mode").is_none());
        assert!(body.get("lateral_contrast").is_none());
        assert!(body["tools"].is_array());
        assert!(body["messages"][0]["tool_calls"].is_array());
        assert!(body["messages"][0]["content"].is_null());
    }
}
