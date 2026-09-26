use std::{
    io::Write,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use ferrite::{
    generate::{chat_prompt, generate},
    kernels::kernels,
    linear::DType,
    model::Model,
    sampler::{Sampler, SamplingParams},
    tokenizer::{StreamDecoder, Tokenizer},
};

#[derive(Parser)]
#[command(version, about = "From-scratch CPU inference for LLMs")]
struct Cli {
    /// Directory containing config.json, model.safetensors and tokenizer.json.
    #[arg(long, short, global = true, default_value = "models/qwen3-0.6b")]
    model: PathBuf,
    /// Maximum context length (KV cache size) in tokens.
    #[arg(long, global = true, default_value_t = 4096)]
    max_seq: usize,
    /// Worker threads (default: number of physical cores).
    #[arg(long, global = true)]
    threads: Option<usize>,
    /// Weight format: f32, bf16, q8 or q4.
    #[arg(long, short, global = true, default_value = "q8")]
    dtype: DType,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Generate a completion for a prompt.
    Run(RunArgs),
    /// Measure prefill and decode throughput with greedy decoding.
    Bench {
        #[arg(long, default_value_t = 128)]
        prompt_len: usize,
        #[arg(long, default_value_t = 128)]
        gen_len: usize,
        /// Repetitions; the median is reported.
        #[arg(long, default_value_t = 5)]
        runs: usize,
    },
    /// Print the token ids for a piece of text.
    Tokenize { text: String },
    /// Perplexity on a text file, optionally with KL divergence against a baseline dtype.
    Eval {
        file: PathBuf,
        /// Tokens per window.
        #[arg(long, default_value_t = 512)]
        ctx: usize,
        /// Number of windows.
        #[arg(long, default_value_t = 4)]
        chunks: usize,
        /// Compare against the same model loaded in this format (e.g. f32).
        #[arg(long)]
        baseline: Option<DType>,
    },
}

#[derive(Args)]
struct RunArgs {
    prompt: String,
    /// Feed the prompt verbatim instead of wrapping it in the chat template.
    #[arg(long)]
    raw: bool,
    /// Let the model emit a <think> block before answering.
    #[arg(long)]
    think: bool,
    #[arg(long)]
    system: Option<String>,
    #[arg(long, short = 'n', default_value_t = 512)]
    max_tokens: usize,
    /// 0 = greedy.
    #[arg(long, short, default_value_t = 0.6)]
    temperature: f32,
    #[arg(long, default_value_t = 20)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long)]
    seed: Option<u64>,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    let tokenizer = Tokenizer::load(cli.model.join("tokenizer.json"))?;

    if let Command::Tokenize { text } = &cli.command {
        let ids = tokenizer.encode(text)?;
        println!("{ids:?}");
        return Ok(());
    }

    if let Some(threads) = cli.threads {
        ferrite::pool::init(threads);
    }
    let start = Instant::now();
    let model = Model::load(&cli.model, cli.max_seq, cli.dtype)?;
    eprintln!(
        "loaded {} as {} in {:.2}s ({:.0} MiB of weights, {} kernels, {} threads)",
        cli.model.display(),
        model.dtype(),
        start.elapsed().as_secs_f64(),
        model.weight_bytes() as f64 / (1 << 20) as f64,
        kernels().isa,
        ferrite::pool::pool().threads(),
    );
    let stop: Vec<u32> = ["<|im_end|>", "<|endoftext|>"]
        .iter()
        .filter_map(|t| tokenizer.token_id(t))
        .collect();

    match cli.command {
        Command::Run(args) => {
            let text = if args.raw {
                args.prompt.clone()
            } else {
                chat_prompt(&args.prompt, args.system.as_deref(), args.think)
            };
            let prompt = tokenizer.encode(&text)?;
            let seed = args.seed.unwrap_or_else(|| {
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64
            });
            let params = SamplingParams {
                temperature: args.temperature,
                top_k: args.top_k,
                top_p: args.top_p,
            };
            let mut sampler = Sampler::new(params, seed);
            let mut session = model.new_session();
            let mut decoder = StreamDecoder::default();
            let mut stdout = std::io::stdout().lock();
            let (stats, _) = generate(
                &mut session,
                &mut sampler,
                &prompt,
                args.max_tokens,
                &stop,
                |id| {
                    let _ = write!(stdout, "{}", decoder.push(&tokenizer, id));
                    let _ = stdout.flush();
                    true
                },
            )?;
            println!();
            eprintln!(
                "prompt: {} tok @ {:.1} tok/s | generated: {} tok @ {:.1} tok/s",
                stats.prompt_tokens,
                stats.prefill_tok_s(),
                stats.generated_tokens,
                stats.decode_tok_s(),
            );
        }
        Command::Bench {
            prompt_len,
            gen_len,
            runs,
        } => {
            // A fixed, deterministic prompt of exactly `prompt_len` tokens.
            let filler = tokenizer.encode(" The quick brown fox jumps over the lazy dog.")?;
            let prompt: Vec<u32> = filler.iter().copied().cycle().take(prompt_len).collect();
            let mut sampler = Sampler::new(SamplingParams::greedy(), 0);
            let mut session = model.new_session();
            let (mut prefill, mut decode) = (Vec::new(), Vec::new());
            // Run 0 is an untimed warm-up that pages the weights in.
            for run in 0..=runs.max(1) {
                session.reset();
                let (stats, _) =
                    generate(&mut session, &mut sampler, &prompt, gen_len, &[], |_| true)?;
                if run > 0 {
                    prefill.push(stats.prefill_tok_s());
                    decode.push(stats.decode_tok_s());
                }
            }
            let (prefill, decode) = (Spread::of(prefill), Spread::of(decode));
            let gb_s = decode.median * model.weight_bytes() as f64 / 1e9;
            println!("prefill: {prefill} tok/s  ({prompt_len} tokens)");
            println!(
                "decode:  {decode} tok/s  ({gen_len} tokens, ~{gb_s:.1} GB/s effective weight bandwidth)"
            );
        }
        Command::Eval {
            file,
            ctx,
            chunks,
            baseline,
        } => {
            let text = std::fs::read_to_string(&file)?;
            let tokens = tokenizer.encode(&text)?;
            let baseline = baseline
                .map(|dtype| Model::load(&cli.model, ctx, dtype))
                .transpose()?;
            let r = ferrite::eval::evaluate(&model, baseline.as_ref(), &tokens, ctx, chunks)?;
            print!(
                "{}: ppl {:.3} over {} tokens",
                model.dtype(),
                r.perplexity,
                r.tokens
            );
            if let (Some(kl), Some(top1)) = (r.kl_divergence, r.top1_agreement) {
                print!(
                    " | KL vs baseline {kl:.5} | top-1 agreement {:.1}%",
                    top1 * 100.0
                );
            }
            println!();
        }
        Command::Tokenize { .. } => unreachable!(),
    }
    Ok(())
}

/// Median and range of repeated measurements.
struct Spread {
    median: f64,
    min: f64,
    max: f64,
}

impl Spread {
    fn of(mut xs: Vec<f64>) -> Self {
        xs.sort_by(f64::total_cmp);
        Self {
            median: xs[xs.len() / 2],
            min: xs[0],
            max: xs[xs.len() - 1],
        }
    }
}

impl std::fmt::Display for Spread {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:>7.1} [{:.1}-{:.1}]", self.median, self.min, self.max)
    }
}
