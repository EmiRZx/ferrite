//! An OpenAI-compatible HTTP API: `POST /v1/chat/completions` (with SSE streaming),
//! `GET /v1/models` and `GET /health`, so existing OpenAI clients work unchanged.

use std::{
    convert::Infallible,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::{SystemTime, UNIX_EPOCH},
};

use axum::{
    Json, Router,
    body::Bytes,
    extract::State,
    http::StatusCode,
    response::{
        IntoResponse, Response,
        sse::{Event as SseEvent, KeepAlive, Sse},
    },
    routing::{get, post},
};
use serde::Deserialize;
use serde_json::{Value, json};

use crate::{
    engine::{Engine, Event, Request, Summary},
    generate::{ChatMessage, FinishReason, chat_prompt},
    sampler::SamplingParams,
};

#[derive(Clone)]
struct AppState {
    engine: Engine,
    model_name: Arc<str>,
}

pub fn router(engine: Engine, model_name: &str) -> Router {
    Router::new()
        .route("/health", get(|| async { "ok" }))
        .route("/v1/models", get(list_models))
        .route("/v1/chat/completions", post(chat_completions))
        .with_state(AppState {
            engine,
            model_name: model_name.into(),
        })
}

pub async fn serve(engine: Engine, model_name: &str, addr: SocketAddr) -> anyhow::Result<()> {
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("serving {model_name} on http://{addr}/v1");
    axum::serve(listener, router(engine, model_name)).await?;
    Ok(())
}

// ----------------------------------------------------------------------------- request

#[derive(Deserialize)]
struct ChatCompletionRequest {
    messages: Vec<Message>,
    #[serde(default)]
    stream: bool,
    stream_options: Option<StreamOptions>,
    max_tokens: Option<usize>,
    max_completion_tokens: Option<usize>,
    temperature: Option<f32>,
    top_p: Option<f32>,
    /// Not in the OpenAI API, but accepted by vLLM and others.
    top_k: Option<usize>,
    seed: Option<u64>,
    stop: Option<Stop>,
    n: Option<usize>,
    /// vLLM-style template options, e.g. `{"enable_thinking": true}` for Qwen3.
    chat_template_kwargs: Option<TemplateKwargs>,
}

#[derive(Deserialize)]
struct Message {
    role: String,
    content: Option<Content>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

#[derive(Deserialize)]
struct ContentPart {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum Stop {
    One(String),
    Many(Vec<String>),
}

#[derive(Deserialize)]
struct StreamOptions {
    #[serde(default)]
    include_usage: bool,
}

#[derive(Deserialize)]
struct TemplateKwargs {
    #[serde(default)]
    enable_thinking: bool,
}

/// Everything a handler needs once the request has been validated.
struct Prepared {
    request: Request,
    stream: bool,
    include_usage: bool,
}

fn prepare(engine: &Engine, body: &[u8]) -> Result<Prepared, ApiError> {
    let req: ChatCompletionRequest = serde_json::from_slice(body)
        .map_err(|e| ApiError::bad_request(format!("invalid request body: {e}")))?;
    if req.n.unwrap_or(1) != 1 {
        return Err(ApiError::bad_request("only n = 1 is supported"));
    }
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must not be empty"));
    }
    let mut messages = Vec::with_capacity(req.messages.len());
    for m in req.messages {
        if !matches!(m.role.as_str(), "system" | "user" | "assistant" | "tool") {
            return Err(ApiError::bad_request(format!(
                "unsupported role {:?}",
                m.role
            )));
        }
        let content = match m.content {
            None => String::new(),
            Some(Content::Text(t)) => t,
            Some(Content::Parts(parts)) => {
                let mut text = String::new();
                for p in parts {
                    match (p.kind.as_str(), p.text) {
                        ("text", Some(t)) => text += &t,
                        (kind, _) => {
                            return Err(ApiError::bad_request(format!(
                                "unsupported content part {kind:?}"
                            )));
                        }
                    }
                }
                text
            }
        };
        messages.push(ChatMessage {
            role: m.role,
            content,
        });
    }

    let thinking = req.chat_template_kwargs.is_some_and(|k| k.enable_thinking);
    let prompt = engine
        .tokenizer()
        .encode(&chat_prompt(&messages, thinking))
        .map_err(|e| ApiError::bad_request(e.to_string()))?;
    let room = engine.max_seq().saturating_sub(prompt.len());
    if room == 0 {
        return Err(ApiError::bad_request(format!(
            "prompt is {} tokens; the context window is {}",
            prompt.len(),
            engine.max_seq()
        )));
    }

    // Qwen3's recommended settings, which differ between thinking and non-thinking mode.
    let (temperature, top_p) = if thinking { (0.6, 0.95) } else { (0.7, 0.8) };
    let sampling = SamplingParams {
        temperature: req.temperature.unwrap_or(temperature),
        top_k: req.top_k.unwrap_or(20),
        top_p: req.top_p.unwrap_or(top_p),
    };
    if !(0.0..=2.0).contains(&sampling.temperature)
        || !(sampling.top_p > 0.0 && sampling.top_p <= 1.0)
    {
        return Err(ApiError::bad_request(
            "temperature must be in [0, 2] and top_p in (0, 1]",
        ));
    }

    let stop = match req.stop {
        None => Vec::new(),
        Some(Stop::One(s)) => vec![s],
        Some(Stop::Many(v)) => v,
    };
    let max_tokens = req
        .max_completion_tokens
        .or(req.max_tokens)
        .unwrap_or(room)
        .min(room);
    Ok(Prepared {
        request: Request {
            prompt,
            max_tokens,
            sampling,
            seed: req.seed.unwrap_or_else(random_u64),
            stop,
        },
        stream: req.stream,
        include_usage: req.stream_options.is_some_and(|o| o.include_usage),
    })
}

// ---------------------------------------------------------------------------- handlers

async fn list_models(State(app): State<AppState>) -> Json<Value> {
    Json(json!({
        "object": "list",
        "data": [{ "id": &*app.model_name, "object": "model", "created": 0, "owned_by": "ferrite" }],
    }))
}

async fn chat_completions(State(app): State<AppState>, body: Bytes) -> Response {
    let prepared = match prepare(&app.engine, &body) {
        Ok(p) => p,
        Err(e) => return e.into_response(),
    };
    let meta = ResponseMeta {
        id: format!("chatcmpl-{:016x}", random_u64()),
        created: unix_time(),
        model: app.model_name.clone(),
    };
    let mut events = app.engine.submit(prepared.request);

    if !prepared.stream {
        let mut content = String::new();
        while let Some(event) = events.recv().await {
            match event {
                Event::Text(t) => content += &t,
                Event::Done(summary) => {
                    return Json(meta.completion(&content, &summary)).into_response();
                }
                Event::Error(e) => return ApiError::server(e).into_response(),
            }
        }
        return ApiError::server("engine closed the request").into_response();
    }

    let include_usage = prepared.include_usage;
    let stream = async_stream::stream! {
        yield sse(meta.chunk(json!({ "role": "assistant", "content": "" }), None));
        while let Some(event) = events.recv().await {
            match event {
                Event::Text(t) => yield sse(meta.chunk(json!({ "content": t }), None)),
                Event::Done(summary) => {
                    yield sse(meta.chunk(json!({}), Some(summary.finish)));
                    if include_usage {
                        let mut usage = meta.chunk(json!({}), None);
                        usage["choices"] = json!([]);
                        usage["usage"] = usage_json(&summary);
                        yield sse(usage);
                    }
                    break;
                }
                Event::Error(e) => {
                    yield sse(json!({ "error": { "message": e, "type": "server_error" } }));
                    break;
                }
            }
        }
        yield Ok::<_, Infallible>(SseEvent::default().data("[DONE]"));
    };
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}

// --------------------------------------------------------------------------- responses

struct ResponseMeta {
    id: String,
    created: u64,
    model: Arc<str>,
}

impl ResponseMeta {
    fn completion(&self, content: &str, summary: &Summary) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion",
            "created": self.created,
            "model": &*self.model,
            "choices": [{
                "index": 0,
                "message": { "role": "assistant", "content": content },
                "finish_reason": finish_reason(summary.finish),
            }],
            "usage": usage_json(summary),
        })
    }

    fn chunk(&self, delta: Value, finish: Option<FinishReason>) -> Value {
        json!({
            "id": self.id,
            "object": "chat.completion.chunk",
            "created": self.created,
            "model": &*self.model,
            "choices": [{ "index": 0, "delta": delta, "finish_reason": finish.map(finish_reason) }],
        })
    }
}

fn finish_reason(f: FinishReason) -> &'static str {
    match f {
        FinishReason::Stop => "stop",
        FinishReason::Length => "length",
    }
}

fn usage_json(summary: &Summary) -> Value {
    let (p, c) = (summary.stats.prompt_tokens, summary.stats.generated_tokens);
    json!({ "prompt_tokens": p, "completion_tokens": c, "total_tokens": p + c })
}

fn sse(value: Value) -> Result<SseEvent, Infallible> {
    Ok(SseEvent::default().data(value.to_string()))
}

/// An error in OpenAI's format: `{"error": {"message", "type", ...}}`.
struct ApiError {
    status: StatusCode,
    kind: &'static str,
    message: String,
}

impl ApiError {
    fn bad_request(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            kind: "invalid_request_error",
            message: message.into(),
        }
    }

    fn server(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            kind: "server_error",
            message: message.into(),
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = json!({ "error": { "message": self.message, "type": self.kind, "param": null, "code": null } });
        (self.status, Json(body)).into_response()
    }
}

fn unix_time() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_secs())
}

/// Unique-enough ids and default seeds: time mixed with a counter (splitmix64).
fn random_u64() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos() as u64);
    let mut z = nanos ^ COUNTER.fetch_add(0x9E37_79B9_7F4A_7C15, Relaxed);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}
