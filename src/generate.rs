use std::time::{Duration, Instant};

use anyhow::Result;

use crate::{model::Session, sampler::Sampler};

#[derive(Debug, Default, Clone)]
pub struct GenStats {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub prefill_time: Duration,
    pub decode_time: Duration,
}

impl GenStats {
    pub fn prefill_tok_s(&self) -> f64 {
        self.prompt_tokens as f64 / self.prefill_time.as_secs_f64()
    }

    /// The first generated token comes out of prefill, so decode covers the rest.
    pub fn decode_tok_s(&self) -> f64 {
        let decoded = self.generated_tokens.saturating_sub(1);
        if decoded == 0 {
            return 0.0;
        }
        decoded as f64 / self.decode_time.as_secs_f64()
    }
}

/// Why generation stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    Stop,
    Length,
}

/// Feeds `prompt` into the session, then samples until a stop token, `max_new`
/// tokens, or `on_token` returning `false`.
pub fn generate(
    session: &mut Session,
    sampler: &mut Sampler,
    prompt: &[u32],
    max_new: usize,
    stop_tokens: &[u32],
    mut on_token: impl FnMut(u32) -> bool,
) -> Result<(GenStats, FinishReason)> {
    let mut stats = GenStats {
        prompt_tokens: prompt.len(),
        ..Default::default()
    };

    let start = Instant::now();
    let mut next = sampler.sample(session.prefill(prompt)?);
    stats.prefill_time = start.elapsed();

    let start = Instant::now();
    let mut reason = FinishReason::Length;
    while stats.generated_tokens < max_new {
        if stop_tokens.contains(&next) {
            reason = FinishReason::Stop;
            break;
        }
        stats.generated_tokens += 1;
        if !on_token(next) {
            reason = FinishReason::Stop;
            break;
        }
        if stats.generated_tokens == max_new {
            break;
        }
        next = sampler.sample(session.forward(next)?);
    }
    stats.decode_time = start.elapsed();
    Ok((stats, reason))
}

/// A single chat turn in the Qwen3 ChatML format.
pub fn chat_prompt(user: &str, system: Option<&str>, thinking: bool) -> String {
    let mut s = String::new();
    if let Some(system) = system {
        s += &format!("<|im_start|>system\n{system}<|im_end|>\n");
    }
    s += &format!("<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n");
    if !thinking {
        s += "<think>\n\n</think>\n\n";
    }
    s
}
