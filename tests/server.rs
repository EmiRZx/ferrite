//! End-to-end tests of the OpenAI-compatible API, calling the router in-process.
//! Needs the model files; skipped when they are absent.

use std::{path::Path, sync::Arc};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use ferrite::{engine::Engine, linear::DType, model::Model, server, tokenizer::Tokenizer};
use serde_json::{Value, json};
use tower::ServiceExt;

fn app() -> Option<Router> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/qwen3-0.6b");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: no model at {}", dir.display());
        return None;
    }
    let tokenizer = Arc::new(Tokenizer::load(dir.join("tokenizer.json")).unwrap());
    let model = Model::load(&dir, 512, DType::Q8).unwrap();
    Some(server::router(
        Engine::start(model, tokenizer),
        "qwen3-0.6b",
    ))
}

async fn post(app: &Router, body: Value) -> (StatusCode, String) {
    let request = Request::post("/v1/chat/completions")
        .header("content-type", "application/json")
        .body(Body::from(body.to_string()))
        .unwrap();
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let bytes = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    (status, String::from_utf8(bytes.to_vec()).unwrap())
}

/// Parses an SSE body into its JSON payloads, checking it ends with `[DONE]`.
fn sse_payloads(body: &str) -> Vec<Value> {
    let data: Vec<&str> = body
        .lines()
        .filter_map(|l| l.strip_prefix("data: "))
        .collect();
    assert_eq!(data.last(), Some(&"[DONE]"), "stream must end with [DONE]");
    data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect()
}

#[tokio::test]
async fn chat_completions_api() {
    let Some(app) = app() else { return };
    let question = json!([{ "role": "user", "content": "What is the capital of France? Answer in one word." }]);

    // Non-streaming, greedy.
    let (status, body) = post(
        &app,
        json!({ "messages": question, "temperature": 0, "max_tokens": 16 }),
    )
    .await;
    assert_eq!(status, StatusCode::OK, "{body}");
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["object"], "chat.completion");
    let content = v["choices"][0]["message"]["content"]
        .as_str()
        .unwrap()
        .to_owned();
    assert!(content.contains("Paris"), "{content}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    let usage = &v["usage"];
    assert_eq!(
        usage["total_tokens"].as_u64().unwrap(),
        usage["prompt_tokens"].as_u64().unwrap() + usage["completion_tokens"].as_u64().unwrap()
    );

    // Streaming the same greedy request yields the same text, chunk by chunk.
    let (status, body) = post(
        &app,
        json!({ "messages": question, "temperature": 0, "max_tokens": 16, "stream": true,
                "stream_options": { "include_usage": true } }),
    )
    .await;
    assert_eq!(status, StatusCode::OK);
    let chunks = sse_payloads(&body);
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let streamed: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(streamed, content);
    assert_eq!(
        chunks[chunks.len() - 2]["choices"][0]["finish_reason"],
        "stop"
    );
    assert_eq!(chunks.last().unwrap()["usage"], *usage);

    // Length limit and stop strings.
    let story = json!([{ "role": "user", "content": "Count from 1 to 20, separated by commas." }]);
    let (_, body) = post(
        &app,
        json!({ "messages": story, "temperature": 0, "max_tokens": 3 }),
    )
    .await;
    let v: Value = serde_json::from_str(&body).unwrap();
    assert_eq!(v["choices"][0]["finish_reason"], "length");
    assert_eq!(v["usage"]["completion_tokens"], 3);

    let (_, body) = post(
        &app,
        json!({ "messages": story, "temperature": 0, "max_tokens": 64, "stop": "5" }),
    )
    .await;
    let v: Value = serde_json::from_str(&body).unwrap();
    let text = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(text.contains('4') && !text.contains('5'), "{text}");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");

    // Errors come back in OpenAI's format.
    for bad in [
        json!({ "messages": [] }),
        json!({ "messages": question, "n": 2 }),
        json!({ "messages": [{ "role": "wizard", "content": "hi" }] }),
        json!({ "messages": question, "temperature": 5 }),
        json!({ "messages": [{ "role": "user", "content": "word ".repeat(600) }] }),
    ] {
        let (status, body) = post(&app, bad).await;
        assert_eq!(status, StatusCode::BAD_REQUEST, "{body}");
        let v: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(v["error"]["type"], "invalid_request_error");
    }
}
