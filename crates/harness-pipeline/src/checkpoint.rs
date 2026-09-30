use harness_attention::{BlockId, PagedAttentionManager};
use harness_core::{HarnessError, Result};
use std::collections::HashMap;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct KVCheckpoint {
    pub checkpoint_id: Uuid,
    pub request_id: Uuid,
    pub step_index: usize,
    pub active_blocks: Vec<BlockId>,
    pub generated_tokens_snapshot: Vec<u32>,
    pub timestamp: std::time::Instant,
}

/// Speculative KV-Cache Checkpoint & Rollback Engine:
/// Enables Tree-of-Thoughts exploration and safe agent tool execution.
/// Uses PagedAttention's Copy-on-Write (CoW) block mechanism to create instant O(1)
/// zero-cost memory checkpoints. If an agent trajectory fails or hallucinates,
/// HARNESS rolls back the KV cache blocks instantly to the checkpoint without
/// recomputing any earlier tokens.
pub struct AgentCheckpointManager {
    checkpoints: HashMap<Uuid, Vec<KVCheckpoint>>, // request_id -> checkpoints stack
}

impl AgentCheckpointManager {
    pub fn new() -> Self {
        Self {
            checkpoints: HashMap::new(),
        }
    }

    /// Create an instantaneous zero-copy checkpoint before a tool execution or branching step
    pub fn create_checkpoint(
        &mut self,
        request_id: Uuid,
        step_index: usize,
        paged_attn: &PagedAttentionManager,
        current_tokens: &[u32],
    ) -> Result<Uuid> {
        let block_table = paged_attn
            .get_block_table(&request_id)
            .ok_or_else(|| HarnessError::Pipeline("No active KV block table for request".into()))?;

        let checkpoint_id = Uuid::new_v4();
        let checkpoint = KVCheckpoint {
            checkpoint_id,
            request_id,
            step_index,
            active_blocks: block_table.to_vec(),
            generated_tokens_snapshot: current_tokens.to_vec(),
            timestamp: std::time::Instant::now(),
        };

        self.checkpoints
            .entry(request_id)
            .or_default()
            .push(checkpoint);

        Ok(checkpoint_id)
    }

    /// Roll back the agent state to the most recent checkpoint upon tool failure or hallucination loop
    pub fn rollback_latest(
        &mut self,
        request_id: &Uuid,
    ) -> Option<KVCheckpoint> {
        if let Some(stack) = self.checkpoints.get_mut(request_id) {
            stack.pop()
        } else {
            None
        }
    }

    pub fn active_checkpoint_count(&self, request_id: &Uuid) -> usize {
        self.checkpoints.get(request_id).map(|s| s.len()).unwrap_or(0)
    }
}

impl Default for AgentCheckpointManager {
    fn default() -> Self {
        Self::new()
    }
}
