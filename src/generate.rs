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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

impl ChatMessage {
    pub fn new(role: &str, content: impl Into<String>) -> Self {
        Self {
            role: role.to_owned(),
            content: content.into(),
        }
    }
}

/// Renders a conversation in Qwen3's ChatML format, ending with the opening of the
/// assistant's turn. Without `thinking`, an empty `<think>` block is pre-filled, which
/// is how Qwen3 disables its reasoning mode.
pub fn chat_prompt(messages: &[ChatMessage], thinking: bool) -> String {
    let mut s = String::new();
    for m in messages {
        s += &format!("<|im_start|>{}\n{}<|im_end|>\n", m.role, m.content);
    }
    s += "<|im_start|>assistant\n";
    if !thinking {
        s += "<think>\n\n</think>\n\n";
    }
    s
}

/// Cuts generated text at the first occurrence of any stop string.
///
/// Text arrives in pieces, so a stop string can span several of them. Anything that
/// could still turn out to be the start of a stop string is held back until it is
/// known not to be.
#[derive(Debug, Default)]
pub struct StopMatcher {
    stops: Vec<String>,
    pending: String,
}

impl StopMatcher {
    pub fn new(stops: &[String]) -> Self {
        Self {
            stops: stops.iter().filter(|s| !s.is_empty()).cloned().collect(),
            pending: String::new(),
        }
    }

    /// Adds `text`; returns the text that is safe to emit and whether a stop string
    /// was hit (in which case nothing from the stop string onward is returned).
    pub fn push(&mut self, text: &str) -> (String, bool) {
        self.pending.push_str(text);
        if let Some(at) = self
            .stops
            .iter()
            .filter_map(|s| self.pending.find(s.as_str()))
            .min()
        {
            let out = self.pending[..at].to_owned();
            self.pending.clear();
            return (out, true);
        }
        // Hold back the longest suffix that is a proper prefix of some stop string.
        let hold = self
            .stops
            .iter()
            .flat_map(|s| {
                let pending = &self.pending;
                (1..s.len().min(pending.len() + 1)).filter(move |&k| {
                    pending.is_char_boundary(pending.len() - k)
                        && s.as_bytes()
                            .starts_with(&pending.as_bytes()[pending.len() - k..])
                })
            })
            .max()
            .unwrap_or(0);
        let out = self.pending[..self.pending.len() - hold].to_owned();
        self.pending.drain(..self.pending.len() - hold);
        (out, false)
    }

    /// Releases any held-back text once generation has ended without a match.
    pub fn finish(&mut self) -> String {
        std::mem::take(&mut self.pending)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(stops: &[&str], pieces: &[&str]) -> (String, bool) {
        let stops: Vec<String> = stops.iter().map(|s| s.to_string()).collect();
        let mut m = StopMatcher::new(&stops);
        let mut out = String::new();
        for p in pieces {
            let (text, hit) = m.push(p);
            out += &text;
            if hit {
                return (out, true);
            }
        }
        out += &m.finish();
        (out, false)
    }

    #[test]
    fn stop_string_split_across_pieces() {
        assert_eq!(
            run(&["END"], &["hello E", "N", "D world"]),
            ("hello ".into(), true)
        );
    }

    #[test]
    fn near_miss_is_released() {
        assert_eq!(run(&["END"], &["the EN", "d"]), ("the ENd".into(), false));
        assert_eq!(
            run(&["END"], &["ends with EN"]),
            ("ends with EN".into(), false)
        );
    }

    #[test]
    fn earliest_of_several_stops_wins() {
        assert_eq!(run(&["b", "a"], &["xxab"]), ("xx".into(), true));
    }

    #[test]
    fn no_stops_passes_everything_through() {
        let mut m = StopMatcher::new(&[]);
        assert_eq!(m.push("abc"), ("abc".into(), false));
    }

    #[test]
    fn multibyte_text_is_never_split() {
        assert_eq!(run(&["ü!"], &["grün", "ü", "x"]), ("grünüx".into(), false));
        assert_eq!(run(&["ğ"], &["dağ"]), ("da".into(), true));
    }

    #[test]
    fn chat_template_matches_qwen3() {
        let p = chat_prompt(
            &[
                ChatMessage::new("system", "Be brief."),
                ChatMessage::new("user", "Hi"),
            ],
            false,
        );
        assert_eq!(
            p,
            "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
    }
}
