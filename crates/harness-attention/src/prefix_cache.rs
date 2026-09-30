use crate::paged_attn::BlockId;
use std::collections::HashMap;

/// Radix tree node for prefix token sequence caching
pub struct RadixNode {
    pub token_seq: Vec<u32>,
    pub blocks: Vec<BlockId>,
    pub children: HashMap<u32, RadixNode>,
}

impl RadixNode {
    pub fn new(token_seq: Vec<u32>, blocks: Vec<BlockId>) -> Self {
        Self {
            token_seq,
            blocks,
            children: HashMap::new(),
        }
    }
}

/// Radix tree prefix cache: enables O(1) KV-cache reuse for shared system prompts & conversation turns
pub struct RadixPrefixCache {
    root: RadixNode,
    total_cached_tokens: usize,
}

impl RadixPrefixCache {
    pub fn new() -> Self {
        Self {
            root: RadixNode::new(vec![], vec![]),
            total_cached_tokens: 0,
        }
    }

    /// Match longest prefix of tokens and return reusable physical blocks
    pub fn match_prefix(&self, tokens: &[u32]) -> (usize, Vec<BlockId>) {
        let mut matched_len = 0;
        let mut reusable_blocks = Vec::new();
        let mut curr = &self.root;

        let mut token_idx = 0;
        while token_idx < tokens.len() {
            let next_tok = tokens[token_idx];
            if let Some(child) = curr.children.get(&next_tok) {
                // Check how much of child.token_seq matches
                let mut match_count = 0;
                for (a, b) in tokens[token_idx..].iter().zip(child.token_seq.iter()) {
                    if a == b {
                        match_count += 1;
                    } else {
                        break;
                    }
                }

                if match_count == child.token_seq.len() {
                    matched_len += match_count;
                    token_idx += match_count;
                    reusable_blocks.extend_from_slice(&child.blocks);
                    curr = child;
                } else {
                    break;
                }
            } else {
                break;
            }
        }

        (matched_len, reusable_blocks)
    }

    /// Insert token sequence and its corresponding physical memory blocks into the prefix tree
    pub fn insert(&mut self, tokens: &[u32], blocks: &[BlockId]) {
        if tokens.is_empty() {
            return;
        }
        let first_token = tokens[0];
        self.root.children.insert(
            first_token,
            RadixNode::new(tokens.to_vec(), blocks.to_vec()),
        );
        self.total_cached_tokens += tokens.len();
    }

    pub fn total_cached_tokens(&self) -> usize {
        self.total_cached_tokens
    }
}

impl Default for RadixPrefixCache {
    fn default() -> Self {
        Self::new()
    }
}
