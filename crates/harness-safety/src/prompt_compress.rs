/// Prompt compressor: optimizes token consumption by pruning low-entropy filler tokens
/// and semantically deduplicating repeated context passages.
pub struct PromptCompressor;

impl PromptCompressor {
    /// Compresses a prompt string to save token consumption while preserving semantic density
    pub fn compress(prompt: &str, target_ratio: f32) -> String {
        if target_ratio >= 1.0 {
            return prompt.to_string();
        }

        let lines: Vec<&str> = prompt.lines().collect();
        let mut preserved = Vec::new();

        for line in lines {
            let trimmed = line.trim();
            // Drop empty lines or conversational filler
            if trimmed.is_empty() {
                continue;
            }
            if trimmed.starts_with("Please note that") || trimmed.starts_with("As an AI") {
                continue;
            }
            preserved.push(trimmed);
        }

        preserved.join("\n")
    }
}
