//! Online Speculative Distillation, Contextual Logit-Bias Fine-Tuning (`OnlineDraftCalibrator`),
//! and Zero-Latency Repository Suffix Automaton (`RepoSuffixDrafter`) for 27B–35B Coder Models.

use std::collections::HashMap;

/// Online speculative distillation adapter that fine-tunes draft token logits in real time
/// from verifier acceptance/rejection feedback using a cross-entropy gradient update.
#[derive(Debug, Clone)]
pub struct OnlineDraftCalibrator {
    /// Learning rate $\eta$ for online logit-bias gradient updates
    pub learning_rate: f32,
    /// L2 decay factor per update to bound adapter norms
    pub weight_decay: f32,
    /// Bigram-conditioned sparse logit bias table: `(prev2_tok, prev1_tok) -> HashMap<token_id, delta_logit>`
    bigram_biases: HashMap<(u32, u32), HashMap<u32, f32>>,
    /// Global unigram token bias adjustments for domain-specific syntax tokens
    unigram_biases: HashMap<u32, f32>,
    /// Total verifier feedback steps processed
    pub total_updates: usize,
    /// Total tokens accepted before and after calibration
    pub accepted_count: usize,
}

impl Default for OnlineDraftCalibrator {
    fn default() -> Self {
        Self::new(0.45, 0.995)
    }
}

impl OnlineDraftCalibrator {
    pub fn new(learning_rate: f32, weight_decay: f32) -> Self {
        Self {
            learning_rate,
            weight_decay,
            bigram_biases: HashMap::new(),
            unigram_biases: HashMap::new(),
            total_updates: 0,
            accepted_count: 0,
        }
    }

    /// Apply learned contextual + unigram logit biases to a draft model's raw logit vector in-place.
    pub fn apply_logit_biases(&self, prev2: u32, prev1: u32, draft_logits: &mut [f32]) {
        for (&tok, &bias) in &self.unigram_biases {
            if let Some(slot) = draft_logits.get_mut(tok as usize) {
                *slot += 0.35 * bias;
            }
        }
        if let Some(ctx_map) = self.bigram_biases.get(&(prev2, prev1)) {
            for (&tok, &bias) in ctx_map {
                if let Some(slot) = draft_logits.get_mut(tok as usize) {
                    *slot += bias;
                }
            }
        }
    }

    /// Perform an online cross-entropy gradient step from a single speculative verification outcome:
    /// - If `draft_tok == target_tok`, reinforces the accepted transition.
    /// - If `draft_tok != target_tok`, boosts `target_tok` by $+\eta (1 - p(y))$ and penalizes
    ///   the rejected `draft_tok` by $-\eta \, p(d)$.
    pub fn observe_verification(
        &mut self,
        prev2: u32,
        prev1: u32,
        draft_tok: u32,
        target_tok: u32,
        draft_prob_on_draft: f32,
        draft_prob_on_target: f32,
    ) {
        self.total_updates += 1;
        let ctx_entry = self.bigram_biases.entry((prev2, prev1)).or_default();

        if draft_tok == target_tok {
            self.accepted_count += 1;
            let boost = 0.15 * self.learning_rate * (1.0 - draft_prob_on_target.clamp(0.0, 1.0));
            let cur = ctx_entry.entry(target_tok).or_insert(0.0);
            *cur = (*cur * self.weight_decay + boost).clamp(-6.0, 6.0);
        } else {
            // Positive gradient on ground-truth verifier token y_t
            let pos_step = self.learning_rate * (1.0 - draft_prob_on_target.clamp(0.0, 1.0));
            let target_bias = ctx_entry.entry(target_tok).or_insert(0.0);
            *target_bias = (*target_bias * self.weight_decay + pos_step).clamp(-6.0, 6.0);

            // Negative gradient on rejected draft token d_t
            let neg_step = self.learning_rate * draft_prob_on_draft.clamp(0.1, 1.0);
            let draft_bias = ctx_entry.entry(draft_tok).or_insert(0.0);
            *draft_bias = (*draft_bias * self.weight_decay - neg_step).clamp(-6.0, 6.0);

            // Mild unigram update for domain vocabulary shifts
            let u_t = self.unigram_biases.entry(target_tok).or_insert(0.0);
            *u_t = (*u_t * self.weight_decay + 0.25 * pos_step).clamp(-3.0, 3.0);
            let u_d = self.unigram_biases.entry(draft_tok).or_insert(0.0);
            *u_d = (*u_d * self.weight_decay - 0.25 * neg_step).clamp(-3.0, 3.0);
        }
    }

    /// Empirical acceptance ratio across all observed verification tokens.
    pub fn empirical_acceptance_rate(&self) -> f32 {
        if self.total_updates == 0 {
            0.0
        } else {
            self.accepted_count as f32 / self.total_updates as f32
        }
    }
}

/// Zero-overhead Repository N-Gram Suffix Drafter (`ngram-mod` / PLD style)
/// that drafts deterministic code continuations from the prompt and active file context.
#[derive(Debug, Clone, Default)]
pub struct RepoSuffixDrafter {
    /// Maps a 3-token suffix `(t_{i-2}, t_{i-1}, t_i)` to the most frequent continuation slice
    trigram_table: HashMap<(u32, u32, u32), Vec<u32>>,
}

impl RepoSuffixDrafter {
    pub fn new() -> Self {
        Self {
            trigram_table: HashMap::new(),
        }
    }

    /// Ingest a tokenized prompt or repository file into the trigram continuation index (`max_draft_len` up to 8).
    pub fn index_tokens(&mut self, tokens: &[u32], max_draft_len: usize) {
        if tokens.len() < 4 || max_draft_len == 0 {
            return;
        }
        for i in 0..(tokens.len() - 3) {
            let key = (tokens[i], tokens[i + 1], tokens[i + 2]);
            let end = (i + 3 + max_draft_len).min(tokens.len());
            let continuation = tokens[i + 3..end].to_vec();
            if !continuation.is_empty() {
                self.trigram_table.insert(key, continuation);
            }
        }
    }

    /// Propose up to `k` draft tokens matching the last 3 context tokens in $O(1)$ time.
    pub fn draft_from_context(&self, context: &[u32], k: usize) -> Option<Vec<u32>> {
        if context.len() < 3 || k == 0 {
            return None;
        }
        let n = context.len();
        let key = (context[n - 3], context[n - 2], context[n - 1]);
        self.trigram_table.get(&key).map(|cont| {
            let take = cont.len().min(k);
            cont[..take].to_vec()
        })
    }
}
