/// xorshift64*: tiny, fast and good enough for token sampling.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.max(1))
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    /// Uniform in `[0, 1)`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

#[derive(Debug, Clone, Copy)]
pub struct SamplingParams {
    /// 0 means greedy decoding.
    pub temperature: f32,
    /// 0 disables top-k filtering.
    pub top_k: usize,
    pub top_p: f32,
}

impl SamplingParams {
    pub fn greedy() -> Self {
        Self {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
        }
    }
}

impl Default for SamplingParams {
    /// Qwen3's recommended settings for thinking mode.
    fn default() -> Self {
        Self {
            temperature: 0.6,
            top_k: 20,
            top_p: 0.95,
        }
    }
}

pub struct Sampler {
    params: SamplingParams,
    rng: Rng,
    candidates: Vec<(f32, u32)>,
}

impl Sampler {
    pub fn new(params: SamplingParams, seed: u64) -> Self {
        Self {
            params,
            rng: Rng::new(seed),
            candidates: Vec::new(),
        }
    }

    pub fn sample(&mut self, logits: &[f32]) -> u32 {
        let p = self.params;
        if p.temperature <= 0.0 {
            return argmax(logits);
        }

        let c = &mut self.candidates;
        c.clear();
        c.extend(logits.iter().enumerate().map(|(i, &l)| (l, i as u32)));
        let by_logit_desc = |a: &(f32, u32), b: &(f32, u32)| b.0.total_cmp(&a.0);
        if p.top_k > 0 && p.top_k < c.len() {
            c.select_nth_unstable_by(p.top_k - 1, by_logit_desc);
            c.truncate(p.top_k);
        }
        c.sort_unstable_by(by_logit_desc);

        // Softmax over the survivors, then keep the smallest prefix with mass >= top_p.
        let max = c[0].0;
        let mut sum = 0.0;
        for (l, _) in c.iter_mut() {
            *l = ((*l - max) / p.temperature).exp();
            sum += *l;
        }
        let mut cumulative = 0.0;
        let mut keep = c.len();
        for (i, (prob, _)) in c.iter_mut().enumerate() {
            *prob /= sum;
            cumulative += *prob;
            if cumulative >= p.top_p {
                keep = i + 1;
                break;
            }
        }
        c.truncate(keep);

        let r = self.rng.next_f32() * c.iter().map(|(prob, _)| prob).sum::<f32>();
        let mut acc = 0.0;
        for &(prob, id) in c.iter() {
            acc += prob;
            if r < acc {
                return id;
            }
        }
        c[c.len() - 1].1
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    x.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map_or(0, |(i, _)| i as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_picks_max() {
        let mut s = Sampler::new(SamplingParams::greedy(), 1);
        assert_eq!(s.sample(&[0.1, 3.0, 2.0]), 1);
    }

    #[test]
    fn top_k_one_is_greedy() {
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 1.0,
                top_k: 1,
                top_p: 1.0,
            },
            7,
        );
        for _ in 0..100 {
            assert_eq!(s.sample(&[0.1, 3.0, 2.9]), 1);
        }
    }

    #[test]
    fn samples_follow_distribution() {
        let mut s = Sampler::new(
            SamplingParams {
                temperature: 1.0,
                top_k: 0,
                top_p: 1.0,
            },
            42,
        );
        let logits = [0.0, (3.0f32).ln()]; // p = [0.25, 0.75]
        let hits = (0..10_000).filter(|_| s.sample(&logits) == 1).count();
        assert!((7_200..7_800).contains(&hits), "{hits}");
    }
}
