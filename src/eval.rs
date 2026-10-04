//! Quality metrics for comparing weight formats.

use anyhow::{Result, ensure};

use crate::{
    model::{Model, PREFILL_CHUNK},
    sampler::argmax,
};

#[derive(Debug, Clone)]
pub struct EvalResult {
    pub tokens: usize,
    pub perplexity: f64,
    /// Mean KL(baseline || model) per token, in nats.
    pub kl_divergence: Option<f64>,
    /// Fraction of positions where the argmax matches the baseline's.
    pub top1_agreement: Option<f64>,
}

/// Scores `chunks` non-overlapping windows of `ctx` tokens. Each window starts from
/// an empty KV cache, and only predictions made at positions `>= score_from` count
/// (the earlier tokens are context only). `score_from = ctx / 2` reproduces
/// llama.cpp's `llama-perplexity`.
pub fn evaluate(
    model: &Model,
    baseline: Option<&Model>,
    tokens: &[u32],
    ctx: usize,
    chunks: usize,
    score_from: usize,
) -> Result<EvalResult> {
    ensure!(ctx >= 2, "context must be at least 2 tokens");
    ensure!(score_from < ctx - 1, "nothing left to score");
    ensure!(
        tokens.len() >= ctx * chunks,
        "need {} tokens, text has {}",
        ctx * chunks,
        tokens.len()
    );

    let vocab = model.config.vocab_size;
    let mut session = model.new_session();
    let mut base_session = baseline.map(Model::new_session);
    let (mut nll, mut kl, mut agree, mut n) = (0.0, 0.0, 0usize, 0usize);
    let mut logp = Vec::new();
    let mut base_logp = Vec::new();

    for chunk in tokens.chunks_exact(ctx).take(chunks) {
        session.reset();
        if let Some(s) = &mut base_session {
            s.reset();
        }
        // Every position's logits are needed, so run batches through `forward_all`.
        let (inputs, targets) = (&chunk[..ctx - 1], &chunk[1..]);
        for (batch, (inputs, targets)) in inputs
            .chunks(PREFILL_CHUNK)
            .zip(targets.chunks(PREFILL_CHUNK))
            .enumerate()
        {
            let logits = session.forward_all(inputs)?;
            let base_logits = match &mut base_session {
                Some(s) => Some(s.forward_all(inputs)?),
                None => None,
            };
            for (i, &target) in targets.iter().enumerate() {
                if batch * PREFILL_CHUNK + i < score_from {
                    continue;
                }
                log_softmax(&logits[i * vocab..][..vocab], &mut logp);
                nll -= logp[target as usize];
                if let Some(base) = base_logits {
                    log_softmax(&base[i * vocab..][..vocab], &mut base_logp);
                    kl += base_logp
                        .iter()
                        .zip(&logp)
                        .map(|(&b, &m)| b.exp() * (b - m))
                        .sum::<f64>();
                    agree += (argmax_f64(&base_logp) == argmax_f64(&logp)) as usize;
                }
                n += 1;
            }
        }
    }

    Ok(EvalResult {
        tokens: n,
        perplexity: (nll / n as f64).exp(),
        kl_divergence: baseline.map(|_| kl / n as f64),
        top1_agreement: baseline.map(|_| agree as f64 / n as f64),
    })
}

fn log_softmax(logits: &[f32], out: &mut Vec<f64>) {
    let max = logits[argmax(logits) as usize] as f64;
    let lse = max
        + logits
            .iter()
            .map(|&l| (l as f64 - max).exp())
            .sum::<f64>()
            .ln();
    out.clear();
    out.extend(logits.iter().map(|&l| l as f64 - lse));
}

fn argmax_f64(x: &[f64]) -> usize {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i)
}
