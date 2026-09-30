use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactedObservation {
    pub original_token_estimate: usize,
    pub compacted_token_estimate: usize,
    pub compacted_text: String,
    pub was_truncated: bool,
    pub spill_pointer: Option<String>,
}

/// Observation Compactor & Semantic Pruning Engine:
/// Eliminates context bloat and "Lost-in-the-Middle" degradation caused by massive
/// raw tool outputs (such as 10,000-line build logs, browser DOM trees, git diffs).
pub struct ObservationCompactor {
    pub max_head_lines: usize,
    pub max_tail_lines: usize,
    pub token_ceiling: usize,
}

impl ObservationCompactor {
    pub fn new(max_head_lines: usize, max_tail_lines: usize, token_ceiling: usize) -> Self {
        Self {
            max_head_lines,
            max_tail_lines,
            token_ceiling,
        }
    }

    /// Compacts raw tool output by extracting critical error signals and head/tail boundaries
    pub fn compact_tool_output(&self, raw_output: &str, tool_name: &str) -> CompactedObservation {
        let original_tokens = raw_output.split_whitespace().count();

        if original_tokens <= self.token_ceiling {
            return CompactedObservation {
                original_token_estimate: original_tokens,
                compacted_token_estimate: original_tokens,
                compacted_text: raw_output.to_string(),
                was_truncated: false,
                spill_pointer: None,
            };
        }

        let lines: Vec<&str> = raw_output.lines().collect();
        let total_lines = lines.len();

        let mut output = String::new();
        output.push_str(&format!(
            "[HARNESS Observation Compactor: Truncated {} lines (~{} tokens) to preserve context memory]\n",
            total_lines, original_tokens
        ));

        // 1. Head lines
        let head_count = self.max_head_lines.min(total_lines);
        for line in &lines[..head_count] {
            output.push_str(line);
            output.push('\n');
        }

        // 2. High-value error lines from middle
        let mut error_lines = Vec::new();
        if total_lines > self.max_head_lines + self.max_tail_lines {
            let middle = &lines[self.max_head_lines..total_lines - self.max_tail_lines];
            for line in middle {
                let lower = line.to_lowercase();
                if lower.contains("error")
                    || lower.contains("fail")
                    || lower.contains("fatal")
                    || lower.contains("panic")
                    || lower.contains("exception")
                {
                    error_lines.push(*line);
                    if error_lines.len() >= 10 {
                        break;
                    }
                }
            }
        }

        if !error_lines.is_empty() {
            output.push_str("\n... [Key Diagnostics Extracted From Middle] ...\n");
            for err in error_lines {
                output.push_str(err);
                output.push('\n');
            }
        }

        output.push_str("\n... [Truncated Middle] ...\n\n");

        // 3. Tail lines
        if total_lines > self.max_head_lines {
            let tail_start = total_lines.saturating_sub(self.max_tail_lines).max(self.max_head_lines);
            for line in &lines[tail_start..] {
                output.push_str(line);
                output.push('\n');
            }
        }

        let compacted_tokens = output.split_whitespace().count();
        let spill_id = format!("spill_{}_{}", tool_name, uuid::Uuid::new_v4());

        CompactedObservation {
            original_token_estimate: original_tokens,
            compacted_token_estimate: compacted_tokens,
            compacted_text: output,
            was_truncated: true,
            spill_pointer: Some(spill_id),
        }
    }
}

impl Default for ObservationCompactor {
    fn default() -> Self {
        Self::new(20, 25, 800)
    }
}
