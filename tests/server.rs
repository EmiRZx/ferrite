//! End-to-end tests of the OpenAI-compatible API, calling the router in-process.
//! Needs the model files; skipped when they are absent.

use std::{path::Path, sync::Arc};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode},
};
use ferrite::{
    engine::{Engine, EngineConfig},
    linear::DType,
    model::Model,
    server,
    tokenizer::Tokenizer,
};
use serde_json::{Value, json};
use tower::ServiceExt;

fn app() -> Option<Router> {
    app_with(EngineConfig {
        max_batch: 4,
        kv_tokens: 2048,
    })
}

fn app_with(config: EngineConfig) -> Option<Router> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("models/qwen3-0.6b");
    if !dir.join("model.safetensors").exists() {
        eprintln!("skipping: no model at {}", dir.display());
        return None;
    }
    let tokenizer = Arc::new(Tokenizer::load(dir.join("tokenizer.json")).unwrap());
    let model = Model::load(&dir, 512, DType::Q8).unwrap();
    Some(server::router(
        Engine::start(model, tokenizer, config),
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

/// Continuous batching must not change results: greedy completions of several
/// requests decoded together in the same steps equal those of each request alone.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_requests_match_sequential_ones() {
    let Some(app) = app() else { return };
    let prompts = [
        "Name the largest planet in the solar system.",
        "Write a haiku about Rust.",
        "What is 17 times 23? Explain briefly.",
        "List three primary colors.",
        "Translate 'good morning' into French and German.",
    ];
    let body = |p: &str| json!({ "messages": [{ "role": "user", "content": p }], "temperature": 0, "max_tokens": 24 });
    let content = |body: &str| -> String {
        let v: Value = serde_json::from_str(body).unwrap();
        v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned()
    };

    let mut alone = Vec::new();
    for p in prompts {
        alone.push(content(&post(&app, body(p)).await.1));
    }

    // Five requests with a batch limit of four: one also has to wait for a free slot.
    let handles: Vec<_> = prompts
        .iter()
        .map(|p| {
            let (app, body) = (app.clone(), body(p));
            tokio::spawn(async move { post(&app, body).await })
        })
        .collect();
    for (handle, want) in handles.into_iter().zip(&alone) {
        let (status, body) = handle.await.unwrap();
        assert_eq!(status, StatusCode::OK);
        assert_eq!(&content(&body), want);
    }
}

/// With a KV cache too small for all running requests, sequences get preempted and
/// later recomputed; the streamed results must be unaffected.
#[tokio::test(flavor = "multi_thread")]
async fn preemption_preserves_results() {
    // 512 tokens = 16 pages; four requests of ~25 + 200 tokens need ~8 pages each.
    let config = EngineConfig {
        max_batch: 4,
        kv_tokens: 512,
    };
    let (Some(small), Some(roomy)) = (app_with(config), app()) else {
        return;
    };
    let body = |i: usize| {
        json!({ "messages": [{ "role": "user", "content": format!("Tell story number {i} about a robot.") }],
                "temperature": 0, "max_tokens": 200, "ignore_eos": true })
    };
    let content = |body: &str| -> String {
        let v: Value = serde_json::from_str(body).unwrap();
        assert_eq!(v["usage"]["completion_tokens"], 200, "{body}");
        v["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .to_owned()
    };

    let mut want = Vec::new();
    for i in 0..4 {
        want.push(content(&post(&roomy, body(i)).await.1));
    }
    let handles: Vec<_> = (0..4)
        .map(|i| {
            let (app, body) = (small.clone(), body(i));
            tokio::spawn(async move { post(&app, body).await })
        })
        .collect();
    for (handle, want) in handles.into_iter().zip(&want) {
        let (status, body) = handle.await.unwrap();
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(&content(&body), want);
    }
}
