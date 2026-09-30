
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
}
