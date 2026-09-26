use std::{
    io::Write,
    path::PathBuf,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use anyhow::Result;
use clap::{Args, Parser, Subcommand};
use ferrite::{
    generate::{chat_prompt, generate},
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
    },
    /// Print the token ids for a piece of text.
    Tokenize { text: String },
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

    let start = Instant::now();
    let model = Model::load(&cli.model, cli.max_seq)?;
    eprintln!(
        "loaded {} in {:.2}s ({:.0} MiB of weights, {} threads)",
        cli.model.display(),
        start.elapsed().as_secs_f64(),
        model.weight_bytes() as f64 / (1 << 20) as f64,
        rayon::current_num_threads(),
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
        } => {
            // A fixed, deterministic prompt of exactly `prompt_len` tokens.
            let filler = tokenizer.encode(" The quick brown fox jumps over the lazy dog.")?;
            let prompt: Vec<u32> = filler.iter().copied().cycle().take(prompt_len).collect();
            let mut sampler = Sampler::new(SamplingParams::greedy(), 0);
            let mut session = model.new_session();
            let (stats, _) = generate(&mut session, &mut sampler, &prompt, gen_len, &[], |_| true)?;
            let bytes_per_token = model.weight_bytes() as f64;
            println!(
                "prefill: {:>8.1} tok/s  ({} tokens)\ndecode:  {:>8.1} tok/s  ({} tokens, ~{:.1} GB/s effective weight bandwidth)",
                stats.prefill_tok_s(),
                stats.prompt_tokens,
                stats.decode_tok_s(),
                stats.generated_tokens,
                stats.decode_tok_s() * bytes_per_token / 1e9,
            );
        }
        Command::Tokenize { .. } => unreachable!(),
    }
    Ok(())
}
