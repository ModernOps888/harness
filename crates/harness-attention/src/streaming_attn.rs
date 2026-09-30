use harness_core::{Result, Tensor};

/// Attention Sinks & StreamingLLM Infinite Context Engine:
/// Solves the "KV Cache Memory Wall" and attention degradation by locking initial tokens
/// (tokens 0..sink_size) which anchor the softmax normalization, combined with a rolling
/// sliding window for local context. Enables infinite-horizon agent loops without OOM.
pub struct AttentionSinkManager {
    pub sink_size: usize,      // Standard: 4 initial sink tokens
    pub window_size: usize,    // Sliding recent window (e.g. 4096 tokens)
    pub total_processed: usize,
}

impl AttentionSinkManager {
    pub fn new(sink_size: usize, window_size: usize) -> Self {
        Self {
            sink_size,
            window_size,
            total_processed: 0,
        }
    }

    /// Calculate active token indices to keep in KV cache:
    /// Always retains [0..sink_size] + [latest - window_size .. latest]
    pub fn active_token_indices(&self, current_seq_len: usize) -> Vec<usize> {
        if current_seq_len <= self.sink_size + self.window_size {
            return (0..current_seq_len).collect();
        }

        let mut indices = Vec::with_capacity(self.sink_size + self.window_size);
        // 1. Initial Attention Sinks
        for i in 0..self.sink_size {
            indices.push(i);
        }
        // 2. Recent Sliding Window
        let window_start = current_seq_len.saturating_sub(self.window_size);
        for i in window_start..current_seq_len {
            indices.push(i);
        }

        indices
    }

    /// Check if KV cache eviction is required
    pub fn should_evict(&self, current_len: usize) -> bool {
        current_len > self.sink_size + self.window_size
    }

    /// Total VRAM savings ratio compared to storing full un-evicted context
    pub fn vram_savings_ratio(&self, full_context_len: usize) -> f32 {
        let active = (self.sink_size + self.window_size).min(full_context_len);
        1.0 - (active as f32 / full_context_len.max(1) as f32)
    }
}
