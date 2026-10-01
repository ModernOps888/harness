use std::collections::VecDeque;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitBreakerState {
    Closed, // Normal execution
    Tripped { reason: String, consecutive_failures: usize },
    Recovering,
}

#[derive(Debug, Clone)]
pub struct AgentActionRecord {
    pub tool_name: String,
    pub arguments_hash: u64,
    pub is_error: bool,
    pub error_snippet: Option<String>,
}

/// Agent Circuit Breaker & Cascading Loop Detector:
/// Prevents catastrophic runaway loops where an agent repeatedly executes failing tools,
/// burning token budgets and compounding hallucinations.
pub struct AgentCircuitBreaker {
    pub max_consecutive_failures: usize,
    pub max_duplicate_actions: usize,
    history: VecDeque<AgentActionRecord>,
    state: CircuitBreakerState,
}

impl AgentCircuitBreaker {
    pub fn new(max_consecutive_failures: usize, max_duplicate_actions: usize) -> Self {
        Self {
            max_consecutive_failures,
            max_duplicate_actions,
            history: VecDeque::with_capacity(16),
            state: CircuitBreakerState::Closed,
        }
    }

    /// Record a tool execution action and check if the circuit breaker should trip
    pub fn record_action(
        &mut self,
        tool_name: &str,
        arguments_json: &str,
        is_error: bool,
        error_msg: Option<&str>,
    ) -> &CircuitBreakerState {
        // Simple hash of arguments
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&arguments_json, &mut hasher);
        let args_hash = std::hash::Hasher::finish(&hasher);

        let record = AgentActionRecord {
            tool_name: tool_name.to_string(),
            arguments_hash: args_hash,
            is_error,
            error_snippet: error_msg.map(|s| s.chars().take(120).collect()),
        };

        self.history.push_back(record);
        let max_history = self.max_consecutive_failures.max(self.max_duplicate_actions).max(16);
        if self.history.len() > max_history {
            self.history.pop_front();
        }

        // 1. Check for consecutive failures
        let mut consecutive_failures = 0;
        for rec in self.history.iter().rev() {
            if rec.is_error {
                consecutive_failures += 1;
            } else {
                break;
            }
        }

        if consecutive_failures >= self.max_consecutive_failures {
            self.state = CircuitBreakerState::Tripped {
                reason: format!(
                    "Consecutive tool failure limit exceeded ({} failures). Intercepting cascading error loop.",
                    consecutive_failures
                ),
                consecutive_failures,
            };
            return &self.state;
        }

        // 2. Check for duplicate action loops (agent retrying same failing tool with same args)
        let mut duplicate_count = 0;
        for rec in self.history.iter().rev() {
            if rec.tool_name == tool_name && rec.arguments_hash == args_hash {
                duplicate_count += 1;
            }
        }

        if duplicate_count >= self.max_duplicate_actions {
            self.state = CircuitBreakerState::Tripped {
                reason: format!(
                    "Duplicate action loop detected: '{}' executed {} times with identical parameters.",
                    tool_name, duplicate_count
                ),
                consecutive_failures,
            };
            return &self.state;
        }

        self.state = CircuitBreakerState::Closed;
        &self.state
    }

    pub fn reset(&mut self) {
        self.state = CircuitBreakerState::Closed;
        self.history.clear();
    }

    pub fn state(&self) -> &CircuitBreakerState {
        &self.state
    }
}
