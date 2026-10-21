//! End-to-end tests of the HARNESS OpenAI proxy against an in-process mock llama-server.
//! They prove the properties VS Code native chat depends on: lossless forwarding of
//! `tools` / `tool_calls` / `role: tool` / `content: null`, streamed tool-call deltas, honest errors.

use axum::extract::State;
use axum::response::sse::{Event, Sse};
use axum::routing::{get, post};
use axum::{Json, Router};
use harness_server::backend::{BackendConfig, BackendType};
use harness_server::routes::create_router;
use harness_server::AppState;
use serde_json::{json, Value};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

type Captured = Arc<Mutex<Vec<Value>>>;

async fn mock_chat(State(cap): State<Captured>, Json(body): Json<Value>) -> axum::response::Response {
    use axum::response::IntoResponse;
    cap.lock().unwrap().push(body.clone());
    let stream = body["stream"].as_bool().unwrap_or(false);
    let wants_tool = body["tools"].is_array();
    if body["max_tokens"] == 999999 {
        return (axum::http::StatusCode::BAD_REQUEST, Json(json!({"error": {"message": "context overflow", "type": "invalid_request_error"}}))).into_response();
    }
    if stream {
        let mut events: Vec<Event> = Vec::new();
        if wants_tool {
            let frames = [
                json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"pa"}}]},"finish_reason":null}]}),
                json!({"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"th\":\"a.rs\"}"}}]},"finish_reason":null}]}),
                json!({"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}),
                json!({"choices":[],"usage":{"prompt_tokens":7,"completion_tokens":9,"total_tokens":16},"timings":{"predicted_per_second":12.5}}),
            ];
            for f in frames { events.push(Event::default().data(f.to_string())); }
        } else {
            for t in ["Hel", "lo"] {
                events.push(Event::default().data(json!({"choices":[{"index":0,"delta":{"content":t},"finish_reason":null}]}).to_string()));
            }
            events.push(Event::default().data(json!({"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}).to_string()));
        }
        events.push(Event::default().data("[DONE]"));
        let s = futures_util::stream::iter(events.into_iter().map(Ok::<_, Infallible>));
        return Sse::new(s).into_response();
    }
    let message = if wants_tool {
        json!({"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":\"a.rs\"}"}}]})
    } else {
        json!({"role":"assistant","content":"hello"})
    };
    Json(json!({
        "id":"x","object":"chat.completion","model":"mock-model",
        "choices":[{"index":0,"message":message,"finish_reason": if wants_tool {"tool_calls"} else {"stop"}}],
        "usage":{"prompt_tokens":5,"completion_tokens":3,"total_tokens":8},
        "timings":{"predicted_per_second":21.0}
    })).into_response()
}

async fn spawn(app: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    format!("http://{}", addr)
}

async fn harness_with_mock() -> (String, Captured) {
    let cap: Captured = Arc::new(Mutex::new(Vec::new()));
    let mock = Router::new()
        .route("/v1/chat/completions", post(mock_chat))
        .route("/v1/models", get(|| async { Json(json!({"object":"list","data":[{"id":"mock-model","object":"model"}]})) }))
        .route("/props", get(|| async { Json(json!({"default_generation_settings":{"n_ctx":32768}})) }))
        .with_state(cap.clone());
    let mock_url = spawn(mock).await;
    let state = AppState::new().with_backend(BackendConfig {
        base_url: mock_url,
        backend_type: BackendType::OpenAICompatible,
        timeout_secs: 30,
        api_key: None,
    });
    (spawn(create_router(state)).await, cap)
}

#[tokio::test]
async fn tools_and_tool_history_are_forwarded_verbatim() {
    let (url, cap) = harness_with_mock().await;
    let tools = json!([{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]);
    let req = json!({
        "model": "auto",
        "messages": [
            {"role":"user","content":[{"type":"text","text":"read it"}]},
            {"role":"assistant","content":null,"tool_calls":[{"id":"call_0","type":"function","function":{"name":"read_file","arguments":"{}"}}]},
            {"role":"tool","tool_call_id":"call_0","content":"fn main(){}"}
        ],
        "tools": tools, "tool_choice": "auto",
        "constrained_mode": "rust", "lateral_contrast": 2.0
    });
    let resp: Value = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&req).send().await.unwrap().json().await.unwrap();

    // Response: tool_calls survive untouched, content stays null, HARNESS metrics are additive.
    assert_eq!(resp["choices"][0]["finish_reason"], "tool_calls");
    assert_eq!(resp["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "read_file");
    assert!(resp["choices"][0]["message"]["content"].is_null());
    assert_eq!(resp["usage"]["prompt_tokens"], 5);
    assert_eq!(resp["usage"]["tokens_per_second"], 21.0);

    // Request as the backend saw it: nothing dropped, HARNESS-only keys stripped, model resolved.
    let seen = cap.lock().unwrap()[0].clone();
    assert_eq!(seen["tools"], tools);
    assert_eq!(seen["tool_choice"], "auto");
    assert_eq!(seen["messages"][1]["tool_calls"][0]["id"], "call_0");
    assert!(seen["messages"][1]["content"].is_null());
    assert_eq!(seen["messages"][2]["role"], "tool");
    assert_eq!(seen["messages"][2]["tool_call_id"], "call_0");
    assert!(seen["messages"][0]["content"].is_array());
    assert!(seen.get("constrained_mode").is_none());
    assert!(seen.get("lateral_contrast").is_none());
    assert_eq!(seen["model"], "mock-model");
}

#[tokio::test]
async fn streamed_tool_call_deltas_pass_through_in_order() {
    let (url, _cap) = harness_with_mock().await;
    let req = json!({"model":"auto","stream":true,"stream_options":{"include_usage":true},
        "messages":[{"role":"user","content":"x"}],
        "tools":[{"type":"function","function":{"name":"read_file","parameters":{"type":"object"}}}]});
    let text = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&req).send().await.unwrap().text().await.unwrap();
    let datas: Vec<&str> = text.lines().filter_map(|l| l.strip_prefix("data:")).map(|d| d.trim()).collect();
    assert_eq!(*datas.last().unwrap(), "[DONE]");
    let mut args = String::new();
    let mut finish = None;
    let mut trailing_harness = None;
    for d in &datas[..datas.len() - 1] {
        let v: Value = serde_json::from_str(d).unwrap();
        if let Some(tc) = v["choices"][0]["delta"]["tool_calls"][0]["function"]["arguments"].as_str() {
            args.push_str(tc);
        }
        if v["choices"][0]["finish_reason"] == "tool_calls" { finish = Some(v.clone()); }
        if v["choices"].as_array().map(|a| a.is_empty()).unwrap_or(false) { trailing_harness = Some(v.clone()); }
    }
    assert_eq!(args, "{\"path\":\"a.rs\"}");
    assert!(finish.is_some());
    let t = trailing_harness.expect("trailing usage chunk forwarded");
    assert_eq!(t["usage"]["prompt_tokens"], 7);
    assert_eq!(t["harness"]["tokens_per_second"], 12.5);
}

#[tokio::test]
async fn plain_stream_forwards_text_deltas() {
    let (url, _cap) = harness_with_mock().await;
    let req = json!({"model":"auto","stream":true,"messages":[{"role":"user","content":"hi"}]});
    let text = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&req).send().await.unwrap().text().await.unwrap();
    let joined: String = text.lines().filter_map(|l| l.strip_prefix("data:")).map(|d| d.trim())
        .filter(|d| *d != "[DONE]")
        .filter_map(|d| serde_json::from_str::<Value>(d).ok())
        .filter_map(|v| v["choices"][0]["delta"]["content"].as_str().map(|s| s.to_string()))
        .collect();
    assert_eq!(joined, "Hello");
}

#[tokio::test]
async fn backend_http_errors_keep_their_status_and_body() {
    let (url, _cap) = harness_with_mock().await;
    let resp = reqwest::Client::new().post(format!("{url}/v1/chat/completions"))
        .json(&json!({"model":"auto","max_tokens":999999,"messages":[{"role":"user","content":"x"}]})).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 400);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["error"]["message"], "context overflow");
}

#[tokio::test]
async fn unreachable_backend_returns_503_openai_error_not_a_fake_answer() {
    let state = AppState::new().with_backend(BackendConfig {
        base_url: "http://127.0.0.1:9".into(),
        backend_type: BackendType::OpenAICompatible,
        timeout_secs: 2,
        api_key: None,
    });
    let url = spawn(create_router(state)).await;
    let resp = reqwest::Client::new().post(format!("{url}/v1/chat/completions"))
        .json(&json!({"model":"auto","messages":[{"role":"user","content":"x"}]})).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 503);
    let v: Value = resp.json().await.unwrap();
    assert_eq!(v["error"]["type"], "backend_unavailable");
    let models = reqwest::Client::new().get(format!("{url}/v1/models")).send().await.unwrap();
    assert_eq!(models.status().as_u16(), 503);
}

#[tokio::test]
async fn models_list_only_what_backend_serves_and_props_report_runtime_ctx() {
    let (url, _cap) = harness_with_mock().await;
    let models: Value = reqwest::get(format!("{url}/v1/models")).await.unwrap().json().await.unwrap();
    let ids: Vec<&str> = models["data"].as_array().unwrap().iter().map(|m| m["id"].as_str().unwrap()).collect();
    assert_eq!(ids, vec!["mock-model"]);
    let props: Value = reqwest::get(format!("{url}/props")).await.unwrap().json().await.unwrap();
    assert_eq!(props["default_generation_settings"]["n_ctx"], 32768);
}

#[tokio::test]
async fn malformed_request_is_a_400() {
    let (url, _cap) = harness_with_mock().await;
    let resp = reqwest::Client::new().post(format!("{url}/v1/chat/completions")).json(&json!({"model":"x"})).send().await.unwrap();
    assert_eq!(resp.status().as_u16(), 400);
}
