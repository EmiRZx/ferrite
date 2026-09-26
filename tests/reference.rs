//! Checks ferrite against outputs dumped from Hugging Face transformers by
//! `scripts/reference.py`. Needs the model files; skipped when they are absent.
//! Run with `cargo test --release`.

use std::path::{Path, PathBuf};

use ferrite::{
    generate::generate,
    linear::DType,
    model::Model,
    sampler::{Sampler, SamplingParams},
    tokenizer::Tokenizer,
};
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

#[derive(Deserialize)]
struct Fixture {
    tokenizer: Vec<TokenizerCase>,
    prompt_ids: Vec<u32>,
    positions: Vec<TopK>,
    greedy: Vec<u32>,
}

#[derive(Deserialize)]
struct TokenizerCase {
    text: String,
    ids: Vec<u32>,
}

#[derive(Deserialize)]
struct TopK {
    ids: Vec<u32>,
    logits: Vec<f32>,
}

fn setup() -> Option<(PathBuf, Fixture)> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_dir = std::env::var_os("FERRITE_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|| root.join("models/qwen3-0.6b"));
    if !model_dir.join("model.safetensors").exists() {
        eprintln!("skipping: no model at {}", model_dir.display());
        return None;
    }
    let text = std::fs::read_to_string(root.join("tests/fixtures/qwen3-0.6b.json")).unwrap();
    Some((model_dir, serde_json::from_str(&text).unwrap()))
}

#[test]
fn tokenizer_matches_hf() {
    let Some((dir, fx)) = setup() else { return };
    let tok = Tokenizer::load(dir.join("tokenizer.json")).unwrap();
    for case in &fx.tokenizer {
        assert_eq!(
            tok.encode(&case.text).unwrap(),
            case.ids,
            "encoding {:?}",
            case.text
        );
        // Encoding NFC-normalizes, so decoding round-trips to the normalized text.
        let normalized: String = case.text.nfc().collect();
        assert_eq!(
            tok.decode(&case.ids),
            normalized,
            "decoding {:?}",
            case.text
        );
    }
}

#[test]
fn logits_and_greedy_match_hf() {
    let Some((dir, fx)) = setup() else { return };
    let model = Model::load(&dir, 256, DType::F32).unwrap();
    let mut session = model.new_session();

    let mut max_err = 0f32;
    for (pos, (&token, expected)) in fx.prompt_ids.iter().zip(&fx.positions).enumerate() {
        let logits = session.forward(token).unwrap();
        let top = ferrite::sampler::argmax(logits);
        assert_eq!(top, expected.ids[0], "argmax differs at position {pos}");
        for (&id, &want) in expected.ids.iter().zip(&expected.logits) {
            max_err = max_err.max((logits[id as usize] - want).abs());
        }
    }
    eprintln!("max |logit - reference| over top-10 at all positions: {max_err:.2e}");
    assert!(max_err < 1e-2, "logits drifted: {max_err}");

    // The batched path (GEMM + causal attention over the whole prompt) must agree too.
    session.reset();
    let vocab = model.config.vocab_size;
    let all = session.forward_all(&fx.prompt_ids).unwrap().to_vec();
    let mut max_err = 0f32;
    for (pos, expected) in fx.positions.iter().enumerate() {
        let logits = &all[pos * vocab..][..vocab];
        assert_eq!(
            ferrite::sampler::argmax(logits),
            expected.ids[0],
            "batched argmax at {pos}"
        );
        for (&id, &want) in expected.ids.iter().zip(&expected.logits) {
            max_err = max_err.max((logits[id as usize] - want).abs());
        }
    }
    eprintln!("batched: max |logit - reference|: {max_err:.2e}");
    assert!(max_err < 1e-2, "batched logits drifted: {max_err}");

    session.reset();
    let mut sampler = Sampler::new(SamplingParams::greedy(), 0);
    let mut out = Vec::new();
    generate(
        &mut session,
        &mut sampler,
        &fx.prompt_ids,
        fx.greedy.len(),
        &[],
        |id| {
            out.push(id);
            true
        },
    )
    .unwrap();
    assert_eq!(out, fx.greedy);
}

/// The batched path (tiled GEMM kernels for quantized weights) must match running the
/// same model one token at a time.
#[test]
fn batched_matches_sequential_for_every_dtype() {
    let Some((dir, fx)) = setup() else { return };
    for dtype in [DType::Bf16, DType::Q8, DType::Q4] {
        let model = Model::load(&dir, 256, dtype).unwrap();
        let vocab = model.config.vocab_size;
        let mut session = model.new_session();
        let batched = session.forward_all(&fx.prompt_ids).unwrap().to_vec();
        session.reset();
        let mut max_err = 0f32;
        for (pos, &token) in fx.prompt_ids.iter().enumerate() {
            let logits = session.forward(token).unwrap();
            let row = &batched[pos * vocab..][..vocab];
            for (a, b) in logits.iter().zip(row) {
                max_err = max_err.max((a - b).abs());
            }
        }
        eprintln!("{dtype}: max |batched - sequential| = {max_err:.2e}");
        assert!(
            max_err < 1e-3,
            "{dtype}: batched and sequential logits differ by {max_err}"
        );
    }
}
