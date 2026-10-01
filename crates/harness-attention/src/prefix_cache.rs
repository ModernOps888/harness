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
        Self::insert_node(&mut self.root, tokens, blocks, &mut self.total_cached_tokens);
    }

    fn insert_node(
        node: &mut RadixNode,
        tokens: &[u32],
        blocks: &[BlockId],
        total_cached: &mut usize,
    ) {
        if tokens.is_empty() {
            return;
        }
        let first_token = tokens[0];
        if let Some(child) = node.children.get_mut(&first_token) {
            let common_len = tokens
                .iter()
                .zip(child.token_seq.iter())
                .take_while(|(&a, &b)| a == b)
                .count();

            if common_len == child.token_seq.len() {
                Self::insert_node(child, &tokens[common_len..], blocks, total_cached);
            } else {
                let split_token = child.token_seq[common_len];
                let mut new_child = RadixNode::new(
                    child.token_seq[common_len..].to_vec(),
                    std::mem::take(&mut child.blocks),
                );
                new_child.children = std::mem::take(&mut child.children);

                child.token_seq.truncate(common_len);
                child.children.insert(split_token, new_child);

                if common_len < tokens.len() {
                    let rem_first = tokens[common_len];
                    let rem_node = RadixNode::new(
                        tokens[common_len..].to_vec(),
                        blocks.to_vec(),
                    );
                    child.children.insert(rem_first, rem_node);
                    *total_cached += tokens.len() - common_len;
                }
            }
        } else {
            node.children.insert(
                first_token,
                RadixNode::new(tokens.to_vec(), blocks.to_vec()),
            );
            *total_cached += tokens.len();
        }
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
