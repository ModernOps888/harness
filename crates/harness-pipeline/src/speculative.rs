
#[derive(Debug, Clone)]
pub struct SpeculativeDraft {
    pub candidate_tokens: Vec<u32>,
    pub confidence_scores: Vec<f32>,
}

/// Speculative decoding engine (EAGLE-3 / Medusa multi-token parallel verification)
pub struct SpeculativeDecoder {
    pub draft_length: usize, // Typically 3-5 draft tokens
    pub accepted_tokens_count: usize,
    pub total_drafted_tokens_count: usize,
}

impl SpeculativeDecoder {
    pub fn new(draft_length: usize) -> Self {
        Self {
            draft_length,
            accepted_tokens_count: 0,
            total_drafted_tokens_count: 0,
        }
    }

    /// Parallel verification of draft candidates against target model's logits
    pub fn verify(
        &mut self,
        drafts: &[u32],
        target_token_predictions: &[u32],
    ) -> (Vec<u32>, usize) {
        let mut accepted = Vec::new();
        let mut accepted_count = 0;

        for (&draft_tok, &target_tok) in drafts.iter().zip(target_token_predictions.iter()) {
            if draft_tok == target_tok {
                accepted.push(draft_tok);
                accepted_count += 1;
            } else {
                // First mismatch: emit the target model's authoritative token and discard subsequent drafts
                accepted.push(target_tok);
                break;
            }
        }

        self.accepted_tokens_count += accepted_count;
        self.total_drafted_tokens_count += drafts.len();

        (accepted, accepted_count)
    }

    /// Current acceptance rate metric (e.g. 0.72 = 72% of draft tokens accepted)
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafted_tokens_count == 0 {
            0.0
        } else {
            self.accepted_tokens_count as f32 / self.total_drafted_tokens_count as f32
        }
    }

    /// Simulate a multi-step speculative execution benchmark with realistic acceptance probability
    pub fn benchmark_simulation(
        &mut self,
        num_steps: usize,
        base_acceptance_prob: f32,
        target_step_ms: f32,
        draft_step_ms: f32,
    ) -> SpeculativeBenchmarkResult {
        let mut total_emitted = 0;
        let mut rng = rand::thread_rng();
        use rand::Rng;

        for _ in 0..num_steps {
            let mut drafts = Vec::with_capacity(self.draft_length);
            let mut targets = Vec::with_capacity(self.draft_length);

            for _ in 0..self.draft_length {
                let tok = rng.gen_range(100..5000);
                drafts.push(tok);
                if rng.gen::<f32>() < base_acceptance_prob {
                    targets.push(tok);
                } else {
                    targets.push(tok + 1); // mismatch
                }
            }

            let (accepted, _count) = self.verify(&drafts, &targets);
            total_emitted += accepted.len();
        }

        let acc_rate = self.acceptance_rate();
        let autoregressive_time_ms = total_emitted as f32 * target_step_ms;
        let spec_step_time = target_step_ms + (self.draft_length as f32 * draft_step_ms);
        let speculative_time_ms = num_steps as f32 * spec_step_time;

        let speedup = if speculative_time_ms > 0.0 {
            autoregressive_time_ms / speculative_time_ms
        } else {
            1.0
        };

        let baseline_tok_per_sec = 1000.0 / target_step_ms;
        let spec_tok_per_sec = baseline_tok_per_sec * speedup;

        SpeculativeBenchmarkResult {
            total_steps: num_steps,
            total_tokens_emitted: total_emitted,
            draft_tokens_count: self.total_drafted_tokens_count,
            accepted_tokens_count: self.accepted_tokens_count,
            acceptance_rate: acc_rate,
            speedup_factor: speedup,
            baseline_tok_per_sec,
            speculative_tok_per_sec: spec_tok_per_sec,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SpeculativeBenchmarkResult {
    pub total_steps: usize,
    pub total_tokens_emitted: usize,
    pub draft_tokens_count: usize,
    pub accepted_tokens_count: usize,
    pub acceptance_rate: f32,
    pub speedup_factor: f32,
    pub baseline_tok_per_sec: f32,
    pub speculative_tok_per_sec: f32,
}

/// Configuration for Entropy-Gated Adaptive Speculative Decoding.
/// Dynamically modulates draft depth K based on Shannon entropy / draft uncertainty,
/// preventing wasted target verification passes on uncertain tokens while expanding depth
/// during low-entropy deterministic spans.
#[derive(Debug, Clone)]
pub struct AdaptiveSpeculativeConfig {
    pub min_depth: usize,
    pub max_depth: usize,
    pub initial_depth: usize,
    pub low_entropy_thresh: f32,   // Below this, draft model is confident -> expand depth
    pub high_entropy_thresh: f32,  // Above this, draft model is uncertain -> contract depth
    pub abort_entropy_thresh: f32, // Critical uncertainty threshold -> immediate clamp to min_depth
}

impl Default for AdaptiveSpeculativeConfig {
    fn default() -> Self {
        Self {
            min_depth: 1,
            max_depth: 8,
            initial_depth: 4,
            low_entropy_thresh: 0.6,
            high_entropy_thresh: 1.8,
            abort_entropy_thresh: 2.4,
        }
    }
}

/// Result of an entropy-gated adaptive verification step
#[derive(Debug, Clone)]
pub struct AdaptiveVerificationResult {
    pub accepted_tokens: Vec<u32>,
    pub accepted_count: usize,
    pub drafted_count: usize,
    pub wasted_tokens: usize,
    pub next_recommended_depth: usize,
}

/// Entropy-Gated Adaptive Speculative Decoder.
/// Eliminates wasted verification passes on offloaded/CPU verifiers while remaining
/// mathematically lossless (identical to target model autoregression distribution).
pub struct AdaptiveSpeculativeDecoder {
    pub config: AdaptiveSpeculativeConfig,
    pub current_depth: usize,
    pub total_accepted_tokens: usize,
    pub total_drafted_tokens: usize,
    pub total_wasted_tokens: usize,
    pub total_steps: usize,
}

impl AdaptiveSpeculativeDecoder {
    pub fn new(config: AdaptiveSpeculativeConfig) -> Self {
        let current_depth = config.initial_depth.clamp(config.min_depth, config.max_depth);
        Self {
            config,
            current_depth,
            total_accepted_tokens: 0,
            total_drafted_tokens: 0,
            total_wasted_tokens: 0,
            total_steps: 0,
        }
    }

    /// Calculate dynamic draft length based on per-token Shannon entropies.
    /// If draft tokens encounter an entropy spike, the draft sequence is pruned early,
    /// preventing costly target verification FLOPs on predictable rejections.
    pub fn determine_effective_draft_len(&self, draft_entropies: &[f32]) -> usize {
        let mut depth = 0;
        for &entropy in draft_entropies.iter().take(self.current_depth) {
            if entropy >= self.config.abort_entropy_thresh {
                // High uncertainty spike: abort drafting further tokens
                depth = depth.max(self.config.min_depth);
                break;
            }
            depth += 1;
        }
        depth.clamp(self.config.min_depth, self.current_depth)
    }

    /// Verify draft tokens against target model predictions with entropy-aware depth adaptation.
    /// Returns accepted tokens, count, wasted tokens, and updates internal adaptation state.
    pub fn verify_step(
        &mut self,
        drafts: &[u32],
        draft_entropies: &[f32],
        target_predictions: &[u32],
    ) -> AdaptiveVerificationResult {
        let effective_len = drafts.len().min(target_predictions.len());
        let mut accepted = Vec::with_capacity(effective_len + 1);
        let mut accepted_count = 0;
        let mut mismatch_occurred = false;

        for i in 0..effective_len {
            let draft_tok = drafts[i];
            let target_tok = target_predictions[i];

            if draft_tok == target_tok {
                accepted.push(draft_tok);
                accepted_count += 1;
            } else {
                // Rejection: emit authoritative target token and prune remaining drafts
                accepted.push(target_tok);
                mismatch_occurred = true;
                break;
            }
        }

        // Wasted tokens are those drafted beyond the first rejection point
        let wasted = drafts.len().saturating_sub(accepted_count + if mismatch_occurred { 1 } else { 0 });

        self.total_accepted_tokens += accepted_count;
        self.total_drafted_tokens += drafts.len();
        self.total_wasted_tokens += wasted;
        self.total_steps += 1;

        // Entropy-gated adaptation of current_depth for subsequent iterations
        let mean_entropy = if draft_entropies.is_empty() {
            1.0
        } else {
            draft_entropies.iter().take(drafts.len()).sum::<f32>() / drafts.len() as f32
        };

        let has_abort_spike = draft_entropies
            .iter()
            .take(drafts.len())
            .any(|&e| e >= self.config.abort_entropy_thresh);

        if has_abort_spike {
            // Immediate collapse to minimum depth
            self.current_depth = self.config.min_depth;
        } else if !mismatch_occurred && mean_entropy <= self.config.low_entropy_thresh {
            // High confidence streak: expand speculative horizon
            self.current_depth = (self.current_depth + 1).min(self.config.max_depth);
        } else if mismatch_occurred || mean_entropy >= self.config.high_entropy_thresh {
            // Mismatch or uncertainty: contract depth to preserve verification FLOPs
            self.current_depth = self.current_depth.saturating_sub(1).max(self.config.min_depth);
        }

        AdaptiveVerificationResult {
            accepted_tokens: accepted,
            accepted_count,
            drafted_count: drafts.len(),
            wasted_tokens: wasted,
            next_recommended_depth: self.current_depth,
        }
    }

    /// Acceptance rate metric across all adaptive steps
    pub fn acceptance_rate(&self) -> f32 {
        if self.total_drafted_tokens == 0 {
            0.0
        } else {
            self.total_accepted_tokens as f32 / self.total_drafted_tokens as f32
        }
    }

    /// Wasted draft token percentage (efficiency metric)
    pub fn wasted_token_ratio(&self) -> f32 {
        if self.total_drafted_tokens == 0 {
            0.0
        } else {
            self.total_wasted_tokens as f32 / self.total_drafted_tokens as f32
        }
    }

    /// Comparative benchmark simulation: Static Depth vs. Entropy-Gated Adaptive Speculative
    pub fn benchmark_adaptive_vs_static(
        &mut self,
        num_steps: usize,
        static_depth: usize,
        target_step_ms: f32,
        draft_step_ms: f32,
    ) -> AdaptiveComparisonReport {
        let mut static_decoder = SpeculativeDecoder::new(static_depth);
        let mut rng = rand::thread_rng();
        use rand::Rng;

        let mut static_wasted = 0;
        let mut static_emitted = 0;
        let mut adaptive_emitted = 0;

        for _ in 0..num_steps {
            // Generate synthetic token distributions with variable entropy regimes
            // (e.g. alternating between low-entropy boilerplate and high-entropy creative reasoning)
            let base_entropy: f32 = rng.gen_range(0.2..2.8);

            // 1. Static Execution
            let mut static_drafts = Vec::with_capacity(static_depth);
            let mut static_targets = Vec::with_capacity(static_depth);
            for _ in 0..static_depth {
                let tok = rng.gen_range(100..5000);
                static_drafts.push(tok);
                let accept_prob = (-0.7 * base_entropy).exp().clamp(0.15, 0.95);
                if rng.gen::<f32>() < accept_prob {
                    static_targets.push(tok);
                } else {
                    static_targets.push(tok + 1);
                }
            }
            let (s_acc, s_count) = static_decoder.verify(&static_drafts, &static_targets);
            static_emitted += s_acc.len();
            let s_mismatch = s_acc.len() < static_drafts.len() + 1 && s_count < static_drafts.len();
            static_wasted += static_drafts.len().saturating_sub(s_count + if s_mismatch { 1 } else { 0 });

            // 2. Adaptive Execution (same underlying token acceptance probabilities)
            let depth = self.current_depth;
            let mut adaptive_drafts = Vec::with_capacity(depth);
            let mut adaptive_entropies = Vec::with_capacity(depth);
            let mut adaptive_targets = Vec::with_capacity(depth);

            for _ in 0..depth {
                let tok = rng.gen_range(100..5000);
                let tok_entropy = (base_entropy + rng.gen_range(-0.2..0.2)).max(0.05);
                adaptive_drafts.push(tok);
                adaptive_entropies.push(tok_entropy);
                let accept_prob = (-0.7 * tok_entropy).exp().clamp(0.15, 0.95);
                if rng.gen::<f32>() < accept_prob {
                    adaptive_targets.push(tok);
                } else {
                    adaptive_targets.push(tok + 1);
                }
            }

            let result = self.verify_step(&adaptive_drafts, &adaptive_entropies, &adaptive_targets);
            adaptive_emitted += result.accepted_tokens.len();
        }

        let static_time_ms = num_steps as f32 * (target_step_ms + static_depth as f32 * draft_step_ms);
        let avg_adaptive_depth = if self.total_steps > 0 {
            self.total_drafted_tokens as f32 / self.total_steps as f32
        } else {
            static_depth as f32
        };
        let adaptive_time_ms = num_steps as f32 * (target_step_ms + avg_adaptive_depth * draft_step_ms);

        let static_tok_per_sec = (static_emitted as f32 / static_time_ms) * 1000.0;
        let adaptive_tok_per_sec = (adaptive_emitted as f32 / adaptive_time_ms) * 1000.0;

        let wasted_reduction_pct = if static_wasted > 0 {
            ((static_wasted as f32 - self.total_wasted_tokens as f32) / static_wasted as f32) * 100.0
        } else {
            0.0
        };

        AdaptiveComparisonReport {
            num_steps,
            static_depth,
            avg_adaptive_depth,
            static_tokens_emitted: static_emitted,
            adaptive_tokens_emitted: adaptive_emitted,
            static_wasted_tokens: static_wasted,
            adaptive_wasted_tokens: self.total_wasted_tokens,
            wasted_tokens_reduction_pct: wasted_reduction_pct.max(0.0),
            static_tok_per_sec,
            adaptive_tok_per_sec,
            speedup_factor: if static_tok_per_sec > 0.0 {
                adaptive_tok_per_sec / static_tok_per_sec
            } else {
                1.0
            },
        }
    }
}

#[derive(Debug, Clone)]
pub struct AdaptiveComparisonReport {
    pub num_steps: usize,
    pub static_depth: usize,
    pub avg_adaptive_depth: f32,
    pub static_tokens_emitted: usize,
    pub adaptive_tokens_emitted: usize,
    pub static_wasted_tokens: usize,
    pub adaptive_wasted_tokens: usize,
    pub wasted_tokens_reduction_pct: f32,
    pub static_tok_per_sec: f32,
    pub adaptive_tok_per_sec: f32,
    pub speedup_factor: f32,
}

