use crate::sampler::{SamplingConfig, TokenSampler};
use harness_core::{Result, Tensor};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct GenerationRequest {
    pub id: Uuid,
    pub prompt_tokens: Vec<u32>,
    pub generated_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub sampling_config: SamplingConfig,
    pub is_finished: bool,
    pub stop_tokens: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct GenerationStepOutput {
    pub request_id: Uuid,
    pub new_token: u32,
    pub is_finished: bool,
}

/// Continuous batching manager:
/// Combines dynamic requests into a single forward step at each token interval,
/// freeing completed requests immediately to maximize throughput (tok/s).
pub struct BatchedInferenceEngine {
    pub active_requests: HashMap<Uuid, GenerationRequest>,
    pub max_batch_size: usize,
    pub total_generated_tokens: usize,
}

impl BatchedInferenceEngine {
    pub fn new(max_batch_size: usize) -> Self {
        Self {
            active_requests: HashMap::new(),
            max_batch_size,
            total_generated_tokens: 0,
        }
    }

    pub fn add_request(&mut self, request: GenerationRequest) -> bool {
        if self.active_requests.len() >= self.max_batch_size {
            return false;
        }
        self.active_requests.insert(request.id, request);
        true
    }

    /// Run one iteration of decode sampling across all active requests in the continuous batch
    pub fn step_batch(
        &mut self,
        logits_by_request: HashMap<Uuid, Tensor>,
        valid_masks: Option<&HashMap<Uuid, Vec<bool>>>,
    ) -> Result<Vec<GenerationStepOutput>> {
        let mut outputs = Vec::new();
        let mut finished_ids = Vec::new();

        for (id, req) in self.active_requests.iter_mut() {
            if let Some(logits) = logits_by_request.get(id) {
                let mask = valid_masks.and_then(|m| m.get(id)).map(|v| v.as_slice());

                let next_token = TokenSampler::sample(
                    logits,
                    &req.sampling_config,
                    mask,
                    &req.generated_tokens,
                )?;

                req.generated_tokens.push(next_token);
                self.total_generated_tokens += 1;

                let reached_max = req.generated_tokens.len() >= req.max_new_tokens;
                let hit_stop = req.stop_tokens.contains(&next_token);

                let is_finished = reached_max || hit_stop;
                req.is_finished = is_finished;

                outputs.push(GenerationStepOutput {
                    request_id: *id,
                    new_token: next_token,
                    is_finished,
                });

                if is_finished {
                    finished_ids.push(*id);
                }
            }
        }

        // Clean up finished requests from continuous batch
        for id in finished_ids {
            self.active_requests.remove(&id);
        }

        Ok(outputs)
    }

    pub fn active_count(&self) -> usize {
        self.active_requests.len()
    }
}
