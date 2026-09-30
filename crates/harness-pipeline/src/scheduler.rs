use std::collections::VecDeque;
use uuid::Uuid;

#[derive(Debug, Clone)]
pub struct QueuedRequest {
    pub id: Uuid,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: usize,
    pub priority: u8,
    pub created_at: std::time::Instant,
}

pub struct RequestScheduler {
    pending: VecDeque<QueuedRequest>,
    max_active_requests: usize,
}

impl RequestScheduler {
    pub fn new(max_active_requests: usize) -> Self {
        Self {
            pending: VecDeque::new(),
            max_active_requests,
        }
    }

    pub fn submit(&mut self, req: QueuedRequest) {
        // High priority requests pushed front, normal requests back
        if req.priority > 10 {
            self.pending.push_front(req);
        } else {
            self.pending.push_back(req);
        }
    }

    pub fn next_batch(&mut self, current_active: usize) -> Vec<QueuedRequest> {
        let available_slots = self.max_active_requests.saturating_sub(current_active);
        let count = available_slots.min(self.pending.len());
        let mut batch = Vec::with_capacity(count);
        for _ in 0..count {
            if let Some(req) = self.pending.pop_front() {
                batch.push(req);
            }
        }
        batch
    }

    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }
}
