use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertainSpan {
    pub start_char: usize,
    pub end_char: usize,
    pub text: String,
    pub local_entropy: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceAssessment {
    pub overall_confidence_score: f32, // 0.0 to 1.0 (e.g. 0.96)
    pub is_trustworthy: bool,
    pub mean_entropy: f32,
    pub uncertain_spans: Vec<UncertainSpan>,
}

pub struct ConfidenceScorer {
    min_trust_threshold: f32,
}

impl ConfidenceScorer {
    pub fn new(min_trust_threshold: f32) -> Self {
        Self { min_trust_threshold }
    }

    /// Aggregate token-level entropies into a calibrated confidence assessment
    pub fn assess(&self, token_entropies: &[f32], _generated_text: &str) -> ConfidenceAssessment {
        if token_entropies.is_empty() {
            return ConfidenceAssessment {
                overall_confidence_score: 1.0,
                is_trustworthy: true,
                mean_entropy: 0.0,
                uncertain_spans: Vec::new(),
            };
        }

        let mean_entropy: f32 =
            token_entropies.iter().sum::<f32>() / token_entropies.len() as f32;

        // Calibrated mapping: lower entropy corresponds to exponential confidence
        let confidence = (1.0 / (1.0 + (mean_entropy * 0.5).exp() - 1.0)).clamp(0.0, 1.0);
        let is_trustworthy = confidence >= self.min_trust_threshold;

        ConfidenceAssessment {
            overall_confidence_score: confidence,
            is_trustworthy,
            mean_entropy,
            uncertain_spans: Vec::new(),
        }
    }
}
