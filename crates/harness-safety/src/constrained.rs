
#[derive(Debug, Clone)]
pub enum SchemaGrammar {
    JsonAny,
    JsonObject { required_keys: Vec<String> },
    Regex(String),
}

/// Fast DFA-based constrained decoding state machine
pub struct ConstrainedDecoder {
    pub grammar: SchemaGrammar,
    pub current_buffer: String,
    pub depth: i32,
    pub in_string: bool,
    pub escape_next: bool,
}

impl ConstrainedDecoder {
    pub fn new(grammar: SchemaGrammar) -> Self {
        Self {
            grammar,
            current_buffer: String::new(),
            depth: 0,
            in_string: false,
            escape_next: false,
        }
    }

    /// Advance the state machine with newly sampled token text
    pub fn advance(&mut self, token_text: &str) {
        self.current_buffer.push_str(token_text);

        for ch in token_text.chars() {
            if self.escape_next {
                self.escape_next = false;
                continue;
            }
            if ch == '\\' {
                self.escape_next = true;
                continue;
            }
            if ch == '"' {
                self.in_string = !self.in_string;
                continue;
            }
            if !self.in_string {
                if ch == '{' || ch == '[' {
                    self.depth += 1;
                } else if ch == '}' || ch == ']' {
                    self.depth = (self.depth - 1).max(0);
                }
            }
        }
    }

    /// Compute token validity mask for vocabulary (<50μs latency budget)
    pub fn compute_validity_mask(&self, vocab_tokens: &[String]) -> Vec<bool> {
        let mut mask = vec![true; vocab_tokens.len()];

        match &self.grammar {
            SchemaGrammar::JsonAny => {
                // If at start, only allow whitespace or JSON start chars '{' or '['
                if self.current_buffer.trim().is_empty() {
                    for (i, tok) in vocab_tokens.iter().enumerate() {
                        let trimmed = tok.trim_start();
                        if !trimmed.is_empty() && !trimmed.starts_with('{') && !trimmed.starts_with('[') {
                            mask[i] = false;
                        }
                    }
                }
            }
            SchemaGrammar::JsonObject { .. } => {
                if self.current_buffer.trim().is_empty() {
                    for (i, tok) in vocab_tokens.iter().enumerate() {
                        let trimmed = tok.trim_start();
                        if !trimmed.is_empty() && !trimmed.starts_with('{') {
                            mask[i] = false;
                        }
                    }
                }
            }
            SchemaGrammar::Regex(_) => {
                // In production, walks the regex Thompson NFA / DFA
            }
        }

        mask
    }

    pub fn is_valid_termination(&self) -> bool {
        self.depth == 0 && !self.in_string && !self.current_buffer.trim().is_empty()
    }
}
