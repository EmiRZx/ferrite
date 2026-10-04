//! The engine thread: owns the model and runs generation requests from a queue,
//! streaming text back over a channel per request.
//!
//! Compute-heavy work stays on this one thread (plus the kernel thread pool), so the
//! async HTTP front end never blocks on it.

use std::{
    sync::{Arc, mpsc},
    thread,
};

use anyhow::{Result, ensure};
use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::{
    generate::{FinishReason, GenStats, StopMatcher, generate},
    model::{Model, Session},
    sampler::{Sampler, SamplingParams},
    tokenizer::{StreamDecoder, Tokenizer},
};

#[derive(Debug, Clone)]
pub struct Request {
    pub prompt: Vec<u32>,
    pub max_tokens: usize,
    pub sampling: SamplingParams,
    pub seed: u64,
    /// Generation ends before the first occurrence of any of these strings.
    pub stop: Vec<String>,
}

#[derive(Debug)]
pub enum Event {
    /// Newly generated text.
    Text(String),
    /// Generation finished; always the last event unless an error occurred.
    Done(Summary),
    Error(String),
}

#[derive(Debug, Clone)]
pub struct Summary {
    pub finish: FinishReason,
    pub stats: GenStats,
}

struct Job {
    request: Request,
    events: UnboundedSender<Event>,
}

/// A cheap, cloneable handle to the engine thread.
#[derive(Clone)]
pub struct Engine {
    jobs: mpsc::Sender<Job>,
    tokenizer: Arc<Tokenizer>,
    max_seq: usize,
}

impl Engine {
    pub fn start(model: Model, tokenizer: Arc<Tokenizer>) -> Self {
        let (jobs, queue) = mpsc::channel();
        let max_seq = model.max_seq;
        let tok = tokenizer.clone();
        thread::Builder::new()
            .name("ferrite-engine".into())
            .spawn(move || run(&model, &tok, queue))
            .expect("spawning engine thread");
        Self {
            jobs,
            tokenizer,
            max_seq,
        }
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Context window in tokens (prompt + completion).
    pub fn max_seq(&self) -> usize {
        self.max_seq
    }

    /// Queues a request; its events arrive on the returned channel. Dropping the
    /// receiver cancels the request.
    pub fn submit(&self, request: Request) -> UnboundedReceiver<Event> {
        let (events, rx) = unbounded_channel();
        if let Err(mpsc::SendError(job)) = self.jobs.send(Job { request, events }) {
            let _ = job.events.send(Event::Error("engine has stopped".into()));
        }
        rx
    }
}

fn run(model: &Model, tokenizer: &Tokenizer, queue: mpsc::Receiver<Job>) {
    let stop_ids: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_id(t))
        .collect();
    let mut session = model.new_session();
    for Job { request, events } in queue {
        if events.is_closed() {
            continue; // cancelled while queued
        }
        session.reset();
        match run_one(&mut session, tokenizer, &stop_ids, &request, &events) {
            Ok(summary) => {
                let s = &summary.stats;
                eprintln!(
                    "request: {} prompt tokens @ {:.0} tok/s, {} generated @ {:.1} tok/s, finish {:?}",
                    s.prompt_tokens,
                    s.prefill_tok_s(),
                    s.generated_tokens,
                    s.decode_tok_s(),
                    summary.finish,
                );
                let _ = events.send(Event::Done(summary));
            }
            Err(e) => {
                let _ = events.send(Event::Error(e.to_string()));
            }
        }
    }
}

fn run_one(
    session: &mut Session,
    tokenizer: &Tokenizer,
    stop_ids: &[u32],
    request: &Request,
    events: &UnboundedSender<Event>,
) -> Result<Summary> {
    ensure!(!request.prompt.is_empty(), "empty prompt");
    let mut sampler = Sampler::new(request.sampling, request.seed);
    let mut decoder = StreamDecoder::default();
    let mut stops = StopMatcher::new(&request.stop);
    let mut hit_stop_string = false;
    let (stats, finish) = generate(
        session,
        &mut sampler,
        &request.prompt,
        request.max_tokens,
        stop_ids,
        |id| {
            let (text, hit) = stops.push(&decoder.push(tokenizer, id));
            // A failed send means the client went away: stop generating.
            if !text.is_empty() && events.send(Event::Text(text)).is_err() {
                return false;
            }
            hit_stop_string = hit;
            !hit
        },
    )?;
    if !hit_stop_string {
        let rest = stops.finish();
        if !rest.is_empty() {
            let _ = events.send(Event::Text(rest));
        }
    }
    Ok(Summary { finish, stats })
}
