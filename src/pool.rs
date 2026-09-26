//! A low-latency thread pool for the decode hot path.
//!
//! A decode step issues ~150 small parallel regions that each take tens of
//! microseconds. A general-purpose pool puts idle workers to sleep, and waking
//! them costs about as much as the work itself. Here workers spin on an epoch
//! counter between jobs (like llama.cpp's threadpool) and only park after being
//! idle for a while, so dispatch takes a few hundred nanoseconds.

use std::{
    cell::UnsafeCell,
    hint,
    ops::Range,
    panic::{AssertUnwindSafe, catch_unwind},
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::*},
    },
    thread::{self, JoinHandle},
};

/// Spin iterations before an idle worker parks (~1-2 ms with `pause`).
const SPIN_LIMIT: u32 = 1 << 16;

type Job = *const (dyn Fn(usize) + Sync + 'static);

struct Shared {
    epoch: AtomicU64,
    remaining: AtomicUsize,
    /// Written by `run` only while every worker is idle; published by the
    /// `epoch` release-store and read after the matching acquire-load.
    job: UnsafeCell<Option<Job>>,
    shutdown: AtomicBool,
    panicked: AtomicBool,
}

// SAFETY: `job` is synchronized through `epoch` / `remaining` as described above.
unsafe impl Sync for Shared {}
unsafe impl Send for Shared {}

pub struct ThreadPool {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    /// Serializes `run` calls from different threads.
    lock: Mutex<()>,
}

impl ThreadPool {
    /// A pool of `threads` threads in total; the caller of `run` acts as thread 0.
    pub fn new(threads: usize) -> Self {
        let shared = Arc::new(Shared {
            epoch: AtomicU64::new(0),
            remaining: AtomicUsize::new(0),
            job: UnsafeCell::new(None),
            shutdown: AtomicBool::new(false),
            panicked: AtomicBool::new(false),
        });
        let workers = (1..threads.max(1))
            .map(|idx| {
                let shared = shared.clone();
                thread::Builder::new()
                    .name(format!("ferrite-{idx}"))
                    .spawn(move || worker(&shared, idx))
                    .expect("spawning worker thread")
            })
            .collect();
        Self {
            shared,
            workers,
            lock: Mutex::new(()),
        }
    }

    pub fn threads(&self) -> usize {
        self.workers.len() + 1
    }

    /// Calls `f(thread_index)` once on every thread and waits for all of them.
    pub fn run(&self, f: &(dyn Fn(usize) + Sync)) {
        let _guard = self.lock.lock().unwrap_or_else(|e| e.into_inner());
        if self.workers.is_empty() {
            return f(0);
        }
        // SAFETY: we erase the lifetime of `f`, but `Wait` below blocks until every
        // worker has finished with it, even if `f(0)` panics.
        let job: Job = unsafe { std::mem::transmute::<*const (dyn Fn(usize) + Sync + '_), Job>(f) };
        unsafe { *self.shared.job.get() = Some(job) };
        self.shared.remaining.store(self.workers.len(), Relaxed);
        self.shared.epoch.fetch_add(1, Release);
        for w in &self.workers {
            w.thread().unpark(); // just an atomic swap unless the worker is parked
        }

        struct Wait<'a>(&'a Shared);
        impl Drop for Wait<'_> {
            fn drop(&mut self) {
                while self.0.remaining.load(Acquire) != 0 {
                    hint::spin_loop();
                }
            }
        }
        let wait = Wait(&self.shared);
        f(0);
        drop(wait);
        if self.shared.panicked.swap(false, Relaxed) {
            panic!("a ferrite worker thread panicked");
        }
    }

    /// Splits `0..n` into chunks of `grain` that threads claim dynamically, so a
    /// slow thread (SMT sibling, lower clock) does not hold everyone else up.
    pub fn par_for(&self, n: usize, grain: usize, f: impl Fn(Range<usize>) + Sync) {
        let grain = grain.max(1);
        if n <= grain {
            return f(0..n);
        }
        let next = AtomicUsize::new(0);
        self.run(&|_| {
            loop {
                let start = next.fetch_add(grain, Relaxed);
                if start >= n {
                    break;
                }
                f(start..(start + grain).min(n));
            }
        });
    }

    /// Fills `out[i] = f(i)` in parallel.
    pub fn par_map(&self, out: &mut [f32], grain: usize, f: impl Fn(usize) -> f32 + Sync) {
        let ptr = SyncPtr(out.as_mut_ptr());
        self.par_for(out.len(), grain, |range| {
            // SAFETY: `par_for` hands out disjoint, in-bounds ranges.
            let chunk =
                unsafe { std::slice::from_raw_parts_mut(ptr.get().add(range.start), range.len()) };
            for (o, i) in chunk.iter_mut().zip(range) {
                *o = f(i);
            }
        });
    }

    /// Fills the row-major `[n, cols]` matrix `out` with `out[t][c] = f(c, t)`, parallel
    /// over columns in chunks of `grain` with `t` innermost. For a GEMM the columns
    /// are weight rows: each one is used for every token while it is hot in L1. With
    /// `grain` = 16, each task writes whole 64-byte lines of each output row.
    pub fn par_matrix(
        &self,
        out: &mut [f32],
        cols: usize,
        grain: usize,
        f: impl Fn(usize, usize) -> f32 + Sync,
    ) {
        assert_eq!(out.len() % cols, 0);
        let n = out.len() / cols;
        let ptr = SyncPtr(out.as_mut_ptr());
        self.par_for(cols, grain, |range| {
            for c in range {
                for t in 0..n {
                    // SAFETY: column ranges are disjoint between tasks and `t * cols + c`
                    // is in bounds.
                    unsafe { *ptr.get().add(t * cols + c) = f(c, t) };
                }
            }
        });
    }

    /// Like [`Self::par_matrix`], but each task gets its whole column range so it can
    /// compute tiles, and stores results through `set(t, c, value)`, which checks
    /// that column `c` belongs to the task.
    pub fn par_matrix_blocks(
        &self,
        out: &mut [f32],
        cols: usize,
        grain: usize,
        f: impl Fn(Range<usize>, &dyn Fn(usize, usize, f32)) + Sync,
    ) {
        assert_eq!(out.len() % cols, 0);
        let n = out.len() / cols;
        let ptr = SyncPtr(out.as_mut_ptr());
        self.par_for(cols, grain, |range| {
            let set = |t: usize, c: usize, v: f32| {
                assert!(
                    t < n && range.contains(&c),
                    "write outside the task's columns"
                );
                // SAFETY: in bounds, and no other task owns column `c`.
                unsafe { *ptr.get().add(t * cols + c) = v };
            };
            f(range.clone(), &set);
        });
    }

    /// Calls `f(i, chunk_i)` in parallel over consecutive `chunk`-sized pieces of `data`
    /// (the last one may be shorter).
    pub fn par_chunks_mut<T: Send>(
        &self,
        data: &mut [T],
        chunk: usize,
        f: impl Fn(usize, &mut [T]) + Sync,
    ) {
        let len = data.len();
        let ptr = SyncPtr(data.as_mut_ptr());
        self.par_for(len.div_ceil(chunk), 1, |range| {
            for i in range {
                let start = i * chunk;
                // SAFETY: chunk `i` is claimed by exactly one thread and lies in bounds.
                let piece = unsafe {
                    std::slice::from_raw_parts_mut(ptr.get().add(start), chunk.min(len - start))
                };
                f(i, piece);
            }
        });
    }

    /// Calls `f(i, a_i, b_i)` in parallel, where `a_i` / `b_i` are the `i`-th chunks
    /// of `a` / `b` of length `ca` / `cb`.
    pub fn par_chunks2<A: Send, B: Send>(
        &self,
        a: &mut [A],
        ca: usize,
        b: &mut [B],
        cb: usize,
        f: impl Fn(usize, &mut [A], &mut [B]) + Sync,
    ) {
        let n = a.len() / ca;
        assert!(a.len() == n * ca && b.len() >= n * cb);
        let (pa, pb) = (SyncPtr(a.as_mut_ptr()), SyncPtr(b.as_mut_ptr()));
        self.par_for(n, 1, |range| {
            for i in range {
                // SAFETY: chunk `i` is claimed by exactly one thread and lies in bounds.
                let (a, b) = unsafe {
                    (
                        std::slice::from_raw_parts_mut(pa.get().add(i * ca), ca),
                        std::slice::from_raw_parts_mut(pb.get().add(i * cb), cb),
                    )
                };
                f(i, a, b);
            }
        });
    }
}

/// A raw pointer that may cross threads; users guarantee disjoint access.
struct SyncPtr<T>(*mut T);

unsafe impl<T: Send> Send for SyncPtr<T> {}
unsafe impl<T: Send> Sync for SyncPtr<T> {}

impl<T> SyncPtr<T> {
    // A method (rather than `.0`) so closures capture the whole Sync wrapper.
    fn get(&self) -> *mut T {
        self.0
    }
}

fn worker(shared: &Shared, idx: usize) {
    let mut seen = 0;
    loop {
        let mut spins = 0;
        loop {
            let epoch = shared.epoch.load(Acquire);
            if epoch != seen {
                seen = epoch;
                break;
            }
            if spins < SPIN_LIMIT {
                spins += 1;
                hint::spin_loop();
            } else {
                thread::park(); // may wake spuriously; the loop re-checks
            }
        }
        if shared.shutdown.load(Acquire) {
            return;
        }
        // SAFETY: published by the epoch release-store; `run` keeps it alive until
        // `remaining` drops to zero.
        let job = unsafe { (*shared.job.get()).expect("job published") };
        if catch_unwind(AssertUnwindSafe(|| unsafe { (*job)(idx) })).is_err() {
            shared.panicked.store(true, Relaxed);
        }
        shared.remaining.fetch_sub(1, Release);
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Release);
        self.shared.epoch.fetch_add(1, Release);
        for w in self.workers.drain(..) {
            w.thread().unpark();
            let _ = w.join();
        }
    }
}

/// The process-wide pool. Sized by [`init`], else `FERRITE_THREADS`, else the
/// number of physical cores.
pub fn pool() -> &'static ThreadPool {
    POOL.get_or_init(|| {
        let threads = std::env::var("FERRITE_THREADS")
            .ok()
            .and_then(|t| t.parse().ok())
            .unwrap_or_else(default_threads);
        ThreadPool::new(threads)
    })
}

/// Sets the global pool size; must be called before the first use of [`pool`].
pub fn init(threads: usize) -> bool {
    POOL.set(ThreadPool::new(threads)).is_ok()
}

/// One thread per physical core. SMT siblings share the core's load ports, so they
/// add nothing to a bandwidth-bound matvec, and with spinning workers any thread
/// the OS preempts stalls the whole barrier.
pub fn default_threads() -> usize {
    num_cpus::get_physical().max(1)
}

static POOL: OnceLock<ThreadPool> = OnceLock::new();

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_for_visits_every_index_once() {
        let pool = ThreadPool::new(4);
        for n in [0, 1, 7, 64, 1000] {
            let hits: Vec<AtomicUsize> = (0..n).map(|_| AtomicUsize::new(0)).collect();
            pool.par_for(n, 3, |range| {
                for i in range {
                    hits[i].fetch_add(1, Relaxed);
                }
            });
            assert!(hits.iter().all(|h| h.load(Relaxed) == 1), "n={n}");
        }
    }

    #[test]
    fn par_map_and_chunks2_write_disjoint_outputs() {
        let pool = ThreadPool::new(3);
        let mut out = vec![0.0; 1001];
        pool.par_map(&mut out, 16, |i| i as f32);
        assert!(out.iter().enumerate().all(|(i, &v)| v == i as f32));

        let mut m = vec![0.0f32; 3 * 37];
        pool.par_matrix(&mut m, 37, 4, |c, t| (t * 100 + c) as f32);
        assert!(
            m.iter()
                .enumerate()
                .all(|(i, &v)| v == ((i / 37) * 100 + i % 37) as f32)
        );

        let mut c = vec![0usize; 23];
        pool.par_chunks_mut(&mut c, 5, |i, chunk| chunk.fill(i));
        assert!(c.iter().enumerate().all(|(j, &v)| v == j / 5));

        let (mut a, mut b) = (vec![0usize; 40], vec![0usize; 25]);
        pool.par_chunks2(&mut a, 8, &mut b, 5, |i, a, b| {
            a.fill(i);
            b.fill(i * 10);
        });
        assert!(a.iter().enumerate().all(|(j, &v)| v == j / 8));
        assert!(b.iter().enumerate().all(|(j, &v)| v == j / 5 * 10));
    }

    #[test]
    fn many_short_jobs() {
        let pool = ThreadPool::new(4);
        let total = AtomicUsize::new(0);
        for _ in 0..10_000 {
            pool.run(&|_| {
                total.fetch_add(1, Relaxed);
            });
        }
        assert_eq!(total.load(Relaxed), 40_000);
    }

    #[test]
    fn worker_panic_propagates_and_pool_survives() {
        let pool = ThreadPool::new(2);
        let r = catch_unwind(AssertUnwindSafe(|| {
            pool.run(&|i| {
                if i == 1 {
                    panic!("boom");
                }
            })
        }));
        assert!(r.is_err());
        let count = AtomicUsize::new(0);
        pool.run(&|_| {
            count.fetch_add(1, Relaxed);
        });
        assert_eq!(count.load(Relaxed), 2);
    }
}
