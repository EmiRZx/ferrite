//! The engine thread: continuous batching over a paged KV cache.
//!
//! Every step is one forward pass that contains the next token of every running
//! sequence plus prompt chunks of sequences still in prefill (vLLM-style chunked
//! prefill). Sequences join and leave between steps, so a long generation never blocks
//! a new request, and all running sequences share each pass over the weights. Decode
//! is memory-bound, so a step with 8 sequences costs little more than a step with one.
//!
//! When KV pages run out, the most recently admitted sequence is preempted: its
//! pages are freed and it waits to be admitted again, at which point its prompt and
//! the tokens generated so far are recomputed in one prefill (vLLM's "recompute"
//! preemption). Text already streamed is never repeated.

use std::{
    collections::VecDeque,
    mem,
    sync::{Arc, mpsc},
    thread,
    time::Instant,
};

use tokio::sync::mpsc::{UnboundedReceiver, UnboundedSender, unbounded_channel};

use crate::{
    generate::{FinishReason, GenStats, StopMatcher},
    kv::{KvPool, PAGE, SeqCache},
    model::{Model, PREFILL_CHUNK, Span, Workspace},
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
    /// Keep generating past end-of-sequence tokens (for benchmarking).
    pub ignore_eos: bool,
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
    /// `prefill_time` here is the time to first token, including queueing.
    pub stats: GenStats,
}

#[derive(Debug, Clone, Copy)]
pub struct EngineConfig {
    /// Most sequences in a step.
    pub max_batch: usize,
    /// KV cache budget in tokens, shared by all sequences.
    pub kv_tokens: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            max_batch: 8,
            kv_tokens: 8192,
        }
    }
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
    pub fn start(model: Model, tokenizer: Arc<Tokenizer>, config: EngineConfig) -> Self {
        let (jobs, queue) = mpsc::channel();
        let max_seq = model.max_seq;
        let tok = tokenizer.clone();
        thread::Builder::new()
            .name("ferrite-engine".into())
            .spawn(move || Scheduler::new(&model, &tok, config).run(queue))
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

/// One request's generation state.
struct Seq {
    events: UnboundedSender<Event>,
    /// Prompt followed by the generated tokens: what the KV cache has to hold.
    tokens: Vec<u32>,
    prompt_len: usize,
    max_tokens: usize,
    ignore_eos: bool,
    cache: SeqCache,
    sampler: Sampler,
    decoder: StreamDecoder,
    stops: StopMatcher,
    /// A stop string matched, so held-back text must not be flushed.
    hit_stop_string: bool,
    generated: usize,
    arrived: Instant,
    first_token: Option<Instant>,
}

impl Seq {
    /// Tokens not yet in the KV cache; the next step feeds (some of) them.
    fn pending(&self) -> usize {
        self.tokens.len() - self.cache.len()
    }

    fn cancelled(&self) -> bool {
        self.events.is_closed()
    }
}

struct Scheduler<'a> {
    model: &'a Model,
    tokenizer: &'a Tokenizer,
    config: EngineConfig,
    pool: KvPool,
    ws: Workspace,
    eos: Vec<u32>,
    waiting: VecDeque<Seq>,
    running: Vec<Seq>,
}

impl<'a> Scheduler<'a> {
    fn new(model: &'a Model, tokenizer: &'a Tokenizer, config: EngineConfig) -> Self {
        // A single sequence must always fit, or preemption could not make progress.
        let kv_tokens = config.kv_tokens.max(model.max_seq);
        Self {
            model,
            tokenizer,
            config,
            pool: KvPool::for_tokens(&model.config, kv_tokens),
            ws: Workspace::new(model, config.max_batch + PREFILL_CHUNK),
            eos: ["<|im_end|>", "<|endoftext|>"]
                .iter()
                .filter_map(|t| tokenizer.token_id(t))
                .collect(),
            waiting: VecDeque::new(),
            running: Vec::new(),
        }
    }

    fn run(mut self, queue: mpsc::Receiver<Job>) {
        eprintln!(
            "engine: up to {} sequences per step, KV cache of {} tokens ({} MiB)",
            self.config.max_batch,
            self.pool.total_pages() * PAGE,
            (self.pool.total_pages() * KvPool::page_bytes(&self.model.config)) >> 20,
        );
        loop {
            // Sleep only when there is nothing to do; otherwise just drain the queue.
            if self.running.is_empty() && self.waiting.is_empty() {
                match queue.recv() {
                    Ok(job) => self.accept(job),
                    Err(_) => return,
                }
            }
            while let Ok(job) = queue.try_recv() {
                self.accept(job);
            }
            self.drop_cancelled();
            self.admit();
            if let Err(e) = self.step() {
                // A failed forward pass poisons every sequence in it.
                for seq in mem::take(&mut self.running) {
                    let _ = seq.events.send(Event::Error(e.to_string()));
                    self.release(seq);
                }
            }
        }
    }

    fn accept(&mut self, Job { request, events }: Job) {
        let pages = (request.prompt.len() + 1).div_ceil(PAGE);
        if request.prompt.is_empty() || pages > self.pool.total_pages() {
            let _ = events.send(Event::Error(
                "prompt is empty or exceeds the KV cache".into(),
            ));
            return;
        }
        let stops = StopMatcher::new(&request.stop);
        self.waiting.push_back(Seq {
            events,
            prompt_len: request.prompt.len(),
            tokens: request.prompt,
            max_tokens: request.max_tokens,
            ignore_eos: request.ignore_eos,
            cache: SeqCache::default(),
            sampler: Sampler::new(request.sampling, request.seed),
            decoder: StreamDecoder::default(),
            stops,
            hit_stop_string: false,
            generated: 0,
            arrived: Instant::now(),
            first_token: None,
        });
    }

    fn drop_cancelled(&mut self) {
        self.waiting.retain(|s| !s.cancelled());
        let (gone, kept): (Vec<Seq>, Vec<Seq>) = mem::take(&mut self.running)
            .into_iter()
            .partition(Seq::cancelled);
        self.running = kept;
        for seq in gone {
            self.release(seq);
        }
    }

    /// Admits waiting sequences in order while there are free slots and enough free
    /// pages for their tokens, keeping one spare page per running sequence so they
    /// can keep growing.
    fn admit(&mut self) {
        while self.running.len() < self.config.max_batch {
            let Some(seq) = self.waiting.front() else {
                break;
            };
            let need = (seq.tokens.len() + 1).div_ceil(PAGE) + self.running.len();
            if need > self.pool.free_pages() {
                break;
            }
            let seq = self.waiting.pop_front().unwrap();
            self.running.push(seq);
        }
    }

    /// Plans and runs one forward pass, then samples for every sequence whose
    /// pending tokens were all fed.
    fn step(&mut self) -> anyhow::Result<()> {
        // Sequences feeding their last pending token (decode) come first, then prompt
        // chunks until the step is full.
        let mut rows_left = self.ws.max_rows();
        let mut plan = vec![0usize; self.running.len()];
        for (i, seq) in self.running.iter().enumerate() {
            if seq.pending() == 1 && rows_left > 0 {
                plan[i] = 1;
                rows_left -= 1;
            }
        }
        for (i, seq) in self.running.iter().enumerate() {
            if plan[i] == 0 && rows_left > 0 {
                plan[i] = seq.pending().min(PREFILL_CHUNK).min(rows_left);
                rows_left -= plan[i];
            }
        }
        self.reserve_pages(&mut plan);
        if plan.iter().all(|&k| k == 0) {
            return Ok(());
        }

        let mut spans = Vec::new();
        let mut owners = Vec::new();
        for (i, seq) in self.running.iter_mut().enumerate() {
            if plan[i] > 0 {
                let start = seq.cache.len();
                spans.push(Span {
                    seq: &mut seq.cache,
                    tokens: &seq.tokens[start..start + plan[i]],
                    all_logits: false,
                });
                owners.push(i);
            }
        }
        self.model
            .forward(&mut self.ws, &mut self.pool, &mut spans)?;
        drop(spans);

        let vocab = self.model.config.vocab_size;
        let mut finished = Vec::new();
        for (span, &i) in owners.iter().enumerate() {
            let seq = &mut self.running[i];
            if seq.pending() == 0 {
                let token = seq.sampler.sample(self.ws.logits(span, vocab));
                if let Some(reason) = on_token(seq, token, self.tokenizer, &self.eos) {
                    finished.push((i, reason));
                }
            }
        }
        for &(i, reason) in finished.iter().rev() {
            let seq = self.running.remove(i);
            self.finish(seq, reason);
        }
        Ok(())
    }

    /// Reserves KV pages for the planned step, oldest sequences first. When the pool
    /// runs dry, the newest running sequence is preempted back to the waiting queue.
    fn reserve_pages(&mut self, plan: &mut Vec<usize>) {
        let mut i = 0;
        while i < self.running.len() {
            let seq = &mut self.running[i];
            let end = seq.cache.len() + plan[i];
            if plan[i] == 0 || self.pool.reserve(&mut seq.cache, end).is_ok() {
                i += 1;
                continue;
            }
            if self.running.len() == 1 {
                // Alone and still out of room: the KV budget is spent, so stop here.
                let seq = self.running.pop().unwrap();
                self.finish(seq, FinishReason::Length);
                plan.clear();
                return;
            }
            let mut victim = self.running.pop().unwrap();
            plan.pop();
            self.pool.release(&mut victim.cache);
            self.waiting.push_front(victim);
            // Retry the same sequence (unless it was the victim) with the freed pages.
        }
    }

    fn finish(&mut self, mut seq: Seq, finish: FinishReason) {
        let rest = seq.stops.finish();
        if !rest.is_empty() && !seq.hit_stop_string {
            let _ = seq.events.send(Event::Text(rest));
        }
        let now = Instant::now();
        let first = seq.first_token.unwrap_or(now);
        let stats = GenStats {
            prompt_tokens: seq.prompt_len,
            generated_tokens: seq.generated,
            prefill_time: first - seq.arrived,
            decode_time: now - first,
        };
        eprintln!(
            "request: {} prompt tokens, first token after {:.0} ms, {} generated @ {:.1} tok/s, finish {:?} ({} running, {} waiting)",
            stats.prompt_tokens,
            stats.prefill_time.as_secs_f64() * 1e3,
            stats.generated_tokens,
            stats.decode_tok_s(),
            finish,
            self.running.len(),
            self.waiting.len(),
        );
        let _ = seq.events.send(Event::Done(Summary { finish, stats }));
        self.release(seq);
    }

    fn release(&mut self, mut seq: Seq) {
        self.pool.release(&mut seq.cache);
    }
}

/// Handles a freshly sampled token; returns why generation ends, if it does.
fn on_token(seq: &mut Seq, token: u32, tokenizer: &Tokenizer, eos: &[u32]) -> Option<FinishReason> {
    if !seq.ignore_eos && eos.contains(&token) {
        return Some(FinishReason::Stop);
    }
    seq.generated += 1;
    seq.first_token.get_or_insert_with(Instant::now);
    let (text, hit) = seq.stops.push(&seq.decoder.push(tokenizer, token));
    seq.hit_stop_string = hit;
    // A failed send means the client went away; it is dropped before the next step.
    if (!text.is_empty() && seq.events.send(Event::Text(text)).is_err()) || hit {
        return Some(FinishReason::Stop);
    }
    if seq.generated >= seq.max_tokens {
        return Some(FinishReason::Length);
    }
    seq.tokens.push(token);
    None
}
