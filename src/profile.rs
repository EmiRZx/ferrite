//! A tiny opt-in profiler: set `FERRITE_PROFILE=1` to accumulate wall time per
//! section of the forward pass. When disabled, [`time`] costs one branch.

use std::{
    fmt::Write,
    sync::{
        LazyLock,
        atomic::{AtomicU64, Ordering::Relaxed},
    },
    time::Instant,
};

#[derive(Debug, Clone, Copy)]
pub enum Section {
    Matmul,
    Quantize,
    Attention,
    Elementwise,
}

const NAMES: [&str; 4] = ["matmul", "quantize", "attention", "elementwise"];

static ENABLED: LazyLock<bool> = LazyLock::new(|| std::env::var_os("FERRITE_PROFILE").is_some());
static TOTAL_NS: [AtomicU64; 4] = [const { AtomicU64::new(0) }; 4];

pub fn enabled() -> bool {
    *ENABLED
}

pub fn time<R>(section: Section, f: impl FnOnce() -> R) -> R {
    if !enabled() {
        return f();
    }
    let start = Instant::now();
    let r = f();
    TOTAL_NS[section as usize].fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
    r
}

/// Adds the time since `start` to `section`.
pub fn add(section: Section, start: Instant) {
    TOTAL_NS[section as usize].fetch_add(start.elapsed().as_nanos() as u64, Relaxed);
}

pub fn reset() {
    TOTAL_NS.iter().for_each(|t| t.store(0, Relaxed));
}

/// One line per section with its share of the profiled time.
pub fn report() -> String {
    let ns: Vec<u64> = TOTAL_NS.iter().map(|t| t.load(Relaxed)).collect();
    let total = ns.iter().sum::<u64>().max(1);
    let mut s = String::new();
    for (name, &t) in NAMES.iter().zip(&ns) {
        let _ = writeln!(
            s,
            "  {name:<12} {:>9.1} ms  {:>5.1}%",
            t as f64 / 1e6,
            100.0 * t as f64 / total as f64
        );
    }
    s
}
