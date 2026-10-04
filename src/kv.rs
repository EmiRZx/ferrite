//! Paged KV cache, in the spirit of vLLM's PagedAttention.
//!
//! All sequences share one pool of fixed-size pages. A page holds [`PAGE`]
//! consecutive positions for every layer and KV head, and a sequence keeps a table of
//! the pages it owns. Sequences of any length can then share memory without
//! fragmentation, and finished sequences hand their pages straight back.
//!
//! Inside a page, each `(layer, head)` gets one attention tile: keys transposed
//! (`[head_dim][PAGE]`) and values row-major (`[PAGE][head_dim]`), as
//! `kernels::attention` expects.

use anyhow::{Result, ensure};

use crate::{
    config::Config,
    kernels::attention::{KvTiles, TILE},
};

/// Positions per page: one attention tile.
pub const PAGE: usize = TILE;

pub struct KvPool {
    k: Vec<f32>,
    v: Vec<f32>,
    layers: usize,
    kv_heads: usize,
    head_dim: usize,
    pages: usize,
    free: Vec<u32>,
}

impl KvPool {
    pub fn new(c: &Config, pages: usize) -> Self {
        let len = pages * Self::page_floats(c);
        Self {
            k: vec![0.0; len],
            v: vec![0.0; len],
            layers: c.num_hidden_layers,
            kv_heads: c.num_key_value_heads,
            head_dim: c.head_dim(),
            pages,
            // Hand out low page ids first.
            free: (0..pages as u32).rev().collect(),
        }
    }

    /// Enough pages for `tokens` positions.
    pub fn for_tokens(c: &Config, tokens: usize) -> Self {
        Self::new(c, tokens.div_ceil(PAGE))
    }

    fn page_floats(c: &Config) -> usize {
        c.num_hidden_layers * c.num_key_value_heads * PAGE * c.head_dim()
    }

    /// Bytes of K and V per page.
    pub fn page_bytes(c: &Config) -> usize {
        2 * Self::page_floats(c) * size_of::<f32>()
    }

    pub fn total_pages(&self) -> usize {
        self.pages
    }

    pub fn free_pages(&self) -> usize {
        self.free.len()
    }

    /// Grows `seq`'s page table so it can hold `len` positions.
    pub fn reserve(&mut self, seq: &mut SeqCache, len: usize) -> Result<()> {
        let need = len.div_ceil(PAGE).saturating_sub(seq.pages.len());
        ensure!(need <= self.free.len(), "KV cache is out of pages");
        for _ in 0..need {
            seq.pages.push(self.free.pop().unwrap());
        }
        Ok(())
    }

    /// Returns all of `seq`'s pages to the pool and empties it.
    pub fn release(&mut self, seq: &mut SeqCache) {
        self.free.extend(seq.pages.drain(..).rev());
        seq.len = 0;
    }

    fn tile_offset(&self, page: u32, layer: usize, head: usize) -> usize {
        ((page as usize * self.layers + layer) * self.kv_heads + head) * PAGE * self.head_dim
    }

    /// Writes one position's key and value for `(layer, head)`.
    pub fn store(
        &mut self,
        seq: &SeqCache,
        pos: usize,
        layer: usize,
        head: usize,
        key: &[f32],
        value: &[f32],
    ) {
        let hd = self.head_dim;
        let base = self.tile_offset(seq.pages[pos / PAGE], layer, head);
        let slot = pos % PAGE;
        for (d, &k) in key.iter().enumerate() {
            self.k[base + d * PAGE + slot] = k;
        }
        self.v[base + slot * hd..][..hd].copy_from_slice(value);
    }

    /// The attention tiles of `seq` for `(layer, head)`.
    pub fn tiles<'a>(&'a self, seq: &'a SeqCache, layer: usize, head: usize) -> SeqTiles<'a> {
        SeqTiles {
            pool: self,
            pages: &seq.pages,
            layer,
            head,
        }
    }
}

/// One sequence's view of the pool: its page table and how many positions are filled.
#[derive(Debug, Default)]
pub struct SeqCache {
    pages: Vec<u32>,
    len: usize,
}

impl SeqCache {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Positions that fit without allocating.
    pub fn capacity(&self) -> usize {
        self.pages.len() * PAGE
    }

    pub fn pages(&self) -> usize {
        self.pages.len()
    }

    /// Records that `n` more positions have been written.
    pub(crate) fn advance(&mut self, n: usize) {
        assert!(self.len + n <= self.capacity());
        self.len += n;
    }
}

pub struct SeqTiles<'a> {
    pool: &'a KvPool,
    pages: &'a [u32],
    layer: usize,
    head: usize,
}

impl KvTiles for SeqTiles<'_> {
    fn tile(&self, i: usize) -> (&[f32], &[f32]) {
        let start = self.pool.tile_offset(self.pages[i], self.layer, self.head);
        let len = PAGE * self.pool.head_dim;
        (&self.pool.k[start..][..len], &self.pool.v[start..][..len])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> Config {
        serde_json::from_value(serde_json::json!({
            "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 2,
            "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4,
            "vocab_size": 10, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
            "max_position_embeddings": 128
        }))
        .unwrap()
    }

    #[test]
    fn pages_are_shared_and_recycled() {
        let c = tiny_config();
        let mut pool = KvPool::new(&c, 4);
        let (mut a, mut b) = (SeqCache::default(), SeqCache::default());
        pool.reserve(&mut a, PAGE + 1).unwrap(); // 2 pages
        pool.reserve(&mut b, PAGE).unwrap(); // 1 page
        assert_eq!((a.pages(), b.pages(), pool.free_pages()), (2, 1, 1));
        assert!(pool.reserve(&mut b, 3 * PAGE).is_err());
        pool.release(&mut a);
        assert_eq!(pool.free_pages(), 3);
        pool.reserve(&mut b, 3 * PAGE).unwrap();
        assert_eq!(pool.free_pages(), 1);
    }

    #[test]
    fn stored_values_come_back_through_tiles() {
        let c = tiny_config();
        let mut pool = KvPool::new(&c, 3);
        let mut seq = SeqCache::default();
        pool.reserve(&mut seq, PAGE + 2).unwrap();
        for pos in 0..PAGE + 2 {
            let key: Vec<f32> = (0..4).map(|d| (pos * 10 + d) as f32).collect();
            let value: Vec<f32> = key.iter().map(|k| -k).collect();
            pool.store(&seq, pos, 1, 0, &key, &value);
        }
        seq.advance(PAGE + 2);
        let tiles = pool.tiles(&seq, 1, 0);
        let (k1, v1) = tiles.tile(1);
        // Position PAGE + 1 is slot 1 of tile 1.
        assert_eq!(k1[3 * PAGE + 1], ((PAGE + 1) * 10 + 3) as f32);
        assert_eq!(v1[4 + 3], -(((PAGE + 1) * 10 + 3) as f32));
    }
}
