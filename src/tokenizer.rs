//! Byte-level BPE tokenizer that reads a Hugging Face `tokenizer.json`.
//!
//! Encoding pipeline (mirrors the HF implementation used by Qwen/GPT-2 style models):
//! 1. split out added tokens (`<|im_start|>`, `<think>`, ...), which map to ids verbatim
//! 2. NFC-normalize the remaining text
//! 3. pre-tokenize with the model's split regex
//! 4. map each piece to bytes and apply BPE merges in rank order

use std::{collections::HashMap, path::Path};

use anyhow::{Context, Result, bail};
use fancy_regex::Regex as FancyRegex;
use regex::Regex;
use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

#[derive(Deserialize)]
struct TokenizerJson {
    added_tokens: Vec<AddedToken>,
    pre_tokenizer: serde_json::Value,
    model: BpeModel,
}

#[derive(Deserialize)]
struct AddedToken {
    id: u32,
    content: String,
}

#[derive(Deserialize)]
struct BpeModel {
    vocab: HashMap<String, u32>,
    merges: Vec<Merge>,
}

/// Newer tokenizer.json files store merges as `["a", "b"]`, older ones as `"a b"`.
#[derive(Deserialize)]
#[serde(untagged)]
enum Merge {
    Pair([String; 2]),
    Joined(String),
}

pub struct Tokenizer {
    /// id -> raw bytes of the token.
    id_to_bytes: Vec<Vec<u8>>,
    /// id of the single-byte token for every byte value.
    byte_to_id: [u32; 256],
    /// (left, right) -> (merge rank, merged id).
    merges: HashMap<(u32, u32), (u32, u32)>,
    added: HashMap<String, u32>,
    added_splitter: Regex,
    pre_tokenizer: FancyRegex,
}

impl Tokenizer {
    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let text =
            std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
        let json: TokenizerJson = serde_json::from_str(&text).context("parsing tokenizer.json")?;

        let unicode_to_byte = byte_level_decoder();
        let token_to_bytes = |tok: &str| -> Result<Vec<u8>> {
            tok.chars()
                .map(|c| {
                    unicode_to_byte
                        .get(&c)
                        .copied()
                        .with_context(|| format!("char {c:?} is not byte-level encoded"))
                })
                .collect()
        };

        let max_id = json
            .model
            .vocab
            .values()
            .chain(json.added_tokens.iter().map(|t| &t.id))
            .copied()
            .max()
            .context("empty vocab")?;
        let mut id_to_bytes = vec![Vec::new(); max_id as usize + 1];
        let mut bytes_to_id = HashMap::with_capacity(json.model.vocab.len());
        for (tok, &id) in &json.model.vocab {
            let bytes = token_to_bytes(tok)?;
            bytes_to_id.insert(bytes.clone(), id);
            id_to_bytes[id as usize] = bytes;
        }

        let mut byte_to_id = [0u32; 256];
        for b in 0..=255u8 {
            byte_to_id[b as usize] = *bytes_to_id
                .get(&vec![b])
                .with_context(|| format!("vocab has no token for byte {b}"))?;
        }

        let mut merges = HashMap::with_capacity(json.model.merges.len());
        for (rank, merge) in json.model.merges.iter().enumerate() {
            let (a, b) = match merge {
                Merge::Pair([a, b]) => (a.as_str(), b.as_str()),
                Merge::Joined(s) => s.split_once(' ').context("malformed merge")?,
            };
            let (a, b) = (token_to_bytes(a)?, token_to_bytes(b)?);
            let merged = [a.as_slice(), b.as_slice()].concat();
            let lookup = |bytes: &Vec<u8>| {
                bytes_to_id
                    .get(bytes)
                    .copied()
                    .context("merge references unknown token")
            };
            merges.insert((lookup(&a)?, lookup(&b)?), (rank as u32, lookup(&merged)?));
        }

        let mut added = HashMap::new();
        for t in &json.added_tokens {
            id_to_bytes[t.id as usize] = t.content.as_bytes().to_vec();
            added.insert(t.content.clone(), t.id);
        }
        // Longest first so that overlapping added tokens prefer the longer match.
        let mut contents: Vec<&str> = added.keys().map(String::as_str).collect();
        contents.sort_by_key(|s| std::cmp::Reverse(s.len()));
        let pattern = contents
            .iter()
            .map(|s| regex::escape(s))
            .collect::<Vec<_>>()
            .join("|");
        let added_splitter = Regex::new(&pattern)?;

        let pre_tokenizer = FancyRegex::new(&find_split_regex(&json.pre_tokenizer)?)?;

        Ok(Self {
            id_to_bytes,
            byte_to_id,
            merges,
            added,
            added_splitter,
            pre_tokenizer,
        })
    }

    pub fn token_id(&self, added_token: &str) -> Option<u32> {
        self.added.get(added_token).copied()
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let mut ids = Vec::new();
        let mut last = 0;
        for m in self.added_splitter.find_iter(text) {
            self.encode_ordinary(&text[last..m.start()], &mut ids)?;
            ids.push(self.added[m.as_str()]);
            last = m.end();
        }
        self.encode_ordinary(&text[last..], &mut ids)?;
        Ok(ids)
    }

    fn encode_ordinary(&self, text: &str, ids: &mut Vec<u32>) -> Result<()> {
        if text.is_empty() {
            return Ok(());
        }
        let normalized: String = text.nfc().collect();
        for piece in self.pre_tokenizer.find_iter(&normalized) {
            self.bpe(piece?.as_str().as_bytes(), ids);
        }
        Ok(())
    }

    /// Repeatedly merges the adjacent pair with the lowest rank until none apply.
    fn bpe(&self, bytes: &[u8], out: &mut Vec<u32>) {
        let mut symbols: Vec<u32> = bytes.iter().map(|&b| self.byte_to_id[b as usize]).collect();
        loop {
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(i, w)| self.merges.get(&(w[0], w[1])).map(|&(r, id)| (r, i, id)))
                .min();
            let Some((_, i, id)) = best else { break };
            symbols[i] = id;
            symbols.remove(i + 1);
        }
        out.extend_from_slice(&symbols);
    }

    /// Raw bytes of a token. Tokens may split a UTF-8 character, see [`StreamDecoder`].
    pub fn token_bytes(&self, id: u32) -> &[u8] {
        self.id_to_bytes.get(id as usize).map_or(&[], Vec::as_slice)
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let bytes: Vec<u8> = ids
            .iter()
            .flat_map(|&id| self.token_bytes(id))
            .copied()
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
}

/// Turns a stream of token ids into text, holding back incomplete UTF-8 sequences.
#[derive(Default)]
pub struct StreamDecoder {
    pending: Vec<u8>,
}

impl StreamDecoder {
    pub fn push(&mut self, tokenizer: &Tokenizer, id: u32) -> String {
        self.pending.extend_from_slice(tokenizer.token_bytes(id));
        match std::str::from_utf8(&self.pending) {
            Ok(s) => {
                let s = s.to_owned();
                self.pending.clear();
                s
            }
            // An incomplete trailing sequence: emit the valid prefix, keep the rest.
            Err(e) if e.error_len().is_none() => {
                let valid = e.valid_up_to();
                let s = String::from_utf8_lossy(&self.pending[..valid]).into_owned();
                self.pending.drain(..valid);
                s
            }
            Err(_) => {
                let s = String::from_utf8_lossy(&self.pending).into_owned();
                self.pending.clear();
                s
            }
        }
    }
}

/// GPT-2's byte-level alphabet: every byte maps to a printable unicode char.
fn byte_level_decoder() -> HashMap<char, u8> {
    let printable = |b: u8| matches!(b, b'!'..=b'~' | 0xA1..=0xAC | 0xAE..=0xFF);
    let mut map = HashMap::with_capacity(256);
    let mut next = 256u32;
    for b in 0..=255u8 {
        let c = if printable(b) {
            b as u32
        } else {
            next += 1;
            next - 1
        };
        map.insert(char::from_u32(c).unwrap(), b);
    }
    map
}

/// Pulls the split regex out of the `pre_tokenizer` section.
fn find_split_regex(pre: &serde_json::Value) -> Result<String> {
    if pre["type"] == "Split"
        && let Some(re) = pre["pattern"]["Regex"].as_str()
    {
        return Ok(re.to_owned());
    }
    if let Some(children) = pre["pretokenizers"].as_array() {
        for child in children {
            if let Ok(re) = find_split_regex(child) {
                return Ok(re);
            }
        }
    }
    bail!("no Split regex found in pre_tokenizer")
}
