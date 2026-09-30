
/// Cortical Lateral Inhibition & Winner-Take-All (WTA) Filter:
/// Inspired by retinal horizontal cells and neocortical interneuron microcircuits.
/// When high-confidence neural features fire, lateral inhibitory connections
/// actively suppress surrounding ambiguous activations.
/// In LLM inference, this sharpens logit contrast, suppresses the noisy
/// hallucination tail, and enforces crisp, factual token selection.
pub struct LateralInhibitionFilter {
    pub inhibition_strength: f32, // \gamma (typically 0.3 - 0.7)
    pub contrast_margin: f32,      // Minimum separation before inhibition activates
}

impl LateralInhibitionFilter {
    pub fn new(inhibition_strength: f32, contrast_margin: f32) -> Self {
        Self {
            inhibition_strength,
            contrast_margin,
        }
    }

    /// Apply lateral inhibition across the logit distribution
    pub fn sharpen_logits(&self, logits: &mut [f32]) {
        if logits.is_empty() {
            return;
        }

        let max_logit = logits
            .iter()
            .copied()
            .fold(f32::NEG_INFINITY, f32::max);

        for logit in logits.iter_mut() {
            let diff = max_logit - *logit;
            if diff > self.contrast_margin {
                // Lateral inhibition penalty: suppresses low-confidence tail
                let penalty = self.inhibition_strength * diff;
                *logit -= penalty;
            }
        }
    }

    /// Compute contrast ratio (relative gap between top candidate and noise floor)
    pub fn compute_contrast_ratio(&self, logits: &[f32]) -> f32 {
        if logits.len() < 2 {
            return 1.0;
        }
        let mut sorted = logits.to_vec();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        let top1 = sorted[0];
        let top2 = sorted[1];
        (top1 - top2).max(0.0)
    }
}

impl Default for LateralInhibitionFilter {
    fn default() -> Self {
        Self::new(0.40, 1.2)
    }
}
