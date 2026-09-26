//! Matrix-vector micro-benchmark: how close does each weight format / ISA get to
//! the machine's memory bandwidth?
//!
//! cargo run --release --example kernels [-- --threads 8]

use std::{hint::black_box, time::Instant};

use anyhow::Result;
use ferrite::{
    kernels::{Isa, Kernels, kernels},
    linear::{DType, Linear},
    quant::Q8Vec,
    sampler::Rng,
};
use rayon::prelude::*;

/// Shapes from Qwen3-0.6B. The LM head (~150-600 MB) streams from DRAM; the small
/// projections fit in L2/L3 and expose per-call overhead instead.
const SHAPES: [(&str, usize, usize); 3] = [
    ("lm_head 151936x1024", 151_936, 1024),
    ("gate_proj 3072x1024", 3072, 1024),
    ("k_proj 1024x1024", 1024, 1024),
];

fn main() -> Result<()> {
    let threads = std::env::args()
        .skip_while(|a| a != "--threads")
        .nth(1)
        .map(|t| t.parse())
        .transpose()?
        .unwrap_or(8);
    ferrite::pool::init(threads);
    rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build_global()?;
    println!("{threads} threads, kernels: {}", kernels().isa);
    println!(
        "available ISAs: {:?}",
        Isa::ALL
            .iter()
            .filter(|i| i.is_supported())
            .collect::<Vec<_>>()
    );
    println!("DRAM read bandwidth: {:.1} GB/s\n", read_bandwidth());

    let mut rng = Rng::new(0);
    for (name, rows, cols) in SHAPES {
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32() - 0.5).collect();
        let x: Vec<f32> = (0..cols).map(|_| rng.next_f32() - 0.5).collect();
        let mut out = vec![0.0; rows];
        let mut xq = Q8Vec::default();
        println!("{name}");
        for dtype in DType::ALL {
            let lin = Linear::from_f32(w.clone(), rows, cols, dtype)?;
            let iters = (2_000_000_000 / lin.bytes()).clamp(5, 5000);
            lin.forward(&mut out, &x, &mut xq);
            let start = Instant::now();
            for _ in 0..iters {
                lin.forward(&mut out, black_box(&x), &mut xq);
            }
            let secs = start.elapsed().as_secs_f64() / iters as f64;
            println!(
                "  {:<5} {:>9.1} us/call  {:>6.1} GB/s",
                dtype.to_string(),
                secs * 1e6,
                lin.bytes() as f64 / secs / 1e9
            );
        }
    }

    // Batched (prefill) matmul: gate_up_proj-sized weights times a 64-token batch.
    let (rows, cols, n) = (6144, 1024, 64);
    println!("\nGEMM {rows}x{cols} weights x {n} tokens (prefill shape):");
    let w: Vec<f32> = (0..rows * cols).map(|_| rng.next_f32() - 0.5).collect();
    let x: Vec<f32> = (0..n * cols).map(|_| rng.next_f32() - 0.5).collect();
    let mut out = vec![0.0; n * rows];
    let mut xq = Q8Vec::default();
    for dtype in DType::ALL {
        let lin = Linear::from_f32(w.clone(), rows, cols, dtype)?;
        lin.forward_batch(&mut out, &x, n, &mut xq);
        let iters = 20;
        let start = Instant::now();
        for _ in 0..iters {
            lin.forward_batch(&mut out, black_box(&x), n, &mut xq);
        }
        let secs = start.elapsed().as_secs_f64() / iters as f64;
        println!(
            "  {:<5} {:>9.1} us/call  {:>6.1} GMAC/s",
            dtype.to_string(),
            secs * 1e6,
            (rows * cols * n) as f64 / secs / 1e9
        );
    }

    println!("\nsingle-thread dot product over one 3072-wide row (in L1/L2):");
    let n = 3072;
    let a: Vec<f32> = (0..n).map(|_| rng.next_f32()).collect();
    for isa in Isa::ALL {
        let Some(k) = Kernels::for_isa(isa) else {
            continue;
        };
        let iters = 200_000;
        let start = Instant::now();
        for _ in 0..iters {
            black_box((k.dot_f32)(black_box(&a), black_box(&a)));
        }
        let ns = start.elapsed().as_secs_f64() * 1e9 / iters as f64;
        println!(
            "  {isa:<6} {ns:>7.1} ns  {:>6.1} GFLOP/s",
            2.0 * n as f64 / ns
        );
    }
    Ok(())
}

/// Parallel streaming read over a buffer much larger than the caches.
fn read_bandwidth() -> f64 {
    let buf = vec![1u64; 1 << 27]; // 1 GiB
    let run = || {
        buf.par_chunks(1 << 16)
            .map(|c| c.iter().fold(0u64, |a, &b| a ^ b))
            .reduce(|| 0, |a, b| a ^ b)
    };
    black_box(run());
    let iters = 5;
    let start = Instant::now();
    for _ in 0..iters {
        black_box(run());
    }
    (buf.len() * 8 * iters) as f64 / start.elapsed().as_secs_f64() / 1e9
}
