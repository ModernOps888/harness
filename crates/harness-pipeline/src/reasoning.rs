use crate::stigmergy::StigmergicTrajectoryManager;
use harness_core::{HarnessError, Result};
use harness_safety::entropy::EntropyDetector;
use harness_safety::lateral_inhibition::LateralInhibitionFilter;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BestOfNConfig {
    /// Number of candidate reasoning branches to sample (typically 3 - 8)
    pub num_candidates: usize,
    /// Maximum tokens per candidate trajectory
    pub max_tokens: usize,
    /// Sampling temperature for exploratory divergence
    pub temperature: f32,
    /// Entropy weighting parameter lambda (penalizes uncertain branches)
    pub entropy_penalty: f32,
    /// Cortical lateral inhibition strength to sharpen logit contrast
    pub inhibition_strength: f32,
    /// Contrast margin for winner-take-all filtering
    pub contrast_margin: f32,
}

impl Default for BestOfNConfig {
    fn default() -> Self {
        Self {
            num_candidates: 4,
            max_tokens: 512,
            temperature: 0.7,
            entropy_penalty: 0.45,
            inhibition_strength: 0.50,
            contrast_margin: 1.2,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateTrajectory {
    pub candidate_id: usize,
    pub tokens: Vec<u32>,
    pub text: String,
    pub mean_entropy: f32,
    pub min_contrast_ratio: f32,
    pub quality_score: f32,
    pub generation_time_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReasoningVerdict {
    pub winning_candidate_id: usize,
    pub champion_text: String,
    pub champion_score: f32,
    pub champion_mean_entropy: f32,
    pub candidates_evaluated: usize,
    pub total_ttc_latency_ms: f64,
    pub pruned_dead_ends: usize,
    pub candidates: Vec<CandidateTrajectory>,
}

/// Test-Time Compute (TTC) Reasoning Engine:
/// Solves complex reasoning, math, and logic tasks by dynamically exploring
/// Best-of-N trajectories scored by Shannon Entropy H(X) and sharpened by
/// cortical lateral inhibition.
pub struct TestTimeReasoningEngine {
    pub config: BestOfNConfig,
    pub stigmergy: StigmergicTrajectoryManager,
    pub inhibitor: LateralInhibitionFilter,
    pub entropy_detector: EntropyDetector,
}

impl TestTimeReasoningEngine {
    pub fn new(config: BestOfNConfig) -> Self {
        let inhibitor = LateralInhibitionFilter::new(config.inhibition_strength, config.contrast_margin);
        let entropy_detector = EntropyDetector::new(2.5);
        let stigmergy = StigmergicTrajectoryManager::default();

        Self {
            config,
            stigmergy,
            inhibitor,
            entropy_detector,
        }
    }

    /// Score a reasoning trajectory based on Shannon entropy and logit contrast
    /// Optimal trajectory has high certainty (low mean entropy) and high contrast margin.
    pub fn score_trajectory(
        &self,
        step_entropies: &[f32],
        step_contrasts: &[f32],
        token_count: usize,
    ) -> f32 {
        if step_entropies.is_empty() || token_count == 0 {
            return 0.0;
        }

        let mean_entropy: f32 = step_entropies.iter().sum::<f32>() / (step_entropies.len() as f32);
        let mean_contrast: f32 = step_contrasts.iter().sum::<f32>() / (step_contrasts.len() as f32);

        // Quality Score: High contrast reward minus entropy uncertainty penalty
        // S = MeanContrast - lambda * MeanEntropy
        let raw_score = mean_contrast - (self.config.entropy_penalty * mean_entropy);
        raw_score.max(0.01)
    }

    /// Select the best reasoning trajectory among evaluated candidates
    pub fn select_champion(
        &mut self,
        candidates: Vec<CandidateTrajectory>,
        total_time_ms: f64,
    ) -> Result<ReasoningVerdict> {
        if candidates.is_empty() {
            return Err(HarnessError::Internal(
                "No candidate trajectories were generated during TTC reasoning".into(),
            ));
        }

        // Rank by quality score descending
        let mut ranked = candidates.clone();
        ranked.sort_by(|a, b| b.quality_score.partial_cmp(&a.quality_score).unwrap_or(std::cmp::Ordering::Equal));

        let winner = ranked[0].clone();

        // Deposit stigmergic pheromone for winning trajectory state
        self.stigmergy.record_transition("prompt_state", &format!("candidate_{}", winner.candidate_id), 1.0);
        self.stigmergy.evaporate();
        let pruned = self.stigmergy.prune_dead_ends(0.2);

        Ok(ReasoningVerdict {
            winning_candidate_id: winner.candidate_id,
            champion_text: winner.text,
            champion_score: winner.quality_score,
            champion_mean_entropy: winner.mean_entropy,
            candidates_evaluated: candidates.len(),
            total_ttc_latency_ms: total_time_ms,
            pruned_dead_ends: pruned,
            candidates,
        })
    }
}
