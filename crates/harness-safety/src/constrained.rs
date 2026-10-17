#[derive(Debug, Clone)]
pub enum SchemaGrammar {
    JsonAny,
    JsonObject { required_keys: Vec<String> },
    Regex(String),
}

/// Fast DFA + Pushdown Automaton (PDA) constrained decoding state machine
pub struct ConstrainedDecoder {
    pub grammar: SchemaGrammar,
    pub current_buffer: String,
    pub depth: i32,
    pub in_string: bool,
    pub escape_next: bool,
    pub delimiter_stack: Vec<char>,
}

impl ConstrainedDecoder {
    pub fn new(grammar: SchemaGrammar) -> Self {
        Self {
            grammar,
            current_buffer: String::new(),
            depth: 0,
            in_string: false,
            escape_next: false,
            delimiter_stack: Vec::new(),
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
            if self.in_string {
                if ch == '\\' {
                    self.escape_next = true;
                } else if ch == '"' {
                    self.in_string = false;
                }
                continue;
            }
            if ch == '"' {
                self.in_string = true;
                continue;
            }
            match ch {
                '{' => {
                    self.depth += 1;
                    self.delimiter_stack.push('}');
                }
                '[' => {
                    self.depth += 1;
                    self.delimiter_stack.push(']');
                }
                '}' | ']' => {
                    if self.delimiter_stack.last().copied() == Some(ch) {
                        self.delimiter_stack.pop();
                    }
                    self.depth = (self.depth - 1).max(0);
                }
                _ => {}
            }
        }
    }

    /// Simulate whether appending `token_text` maintains valid pushdown delimiter balance
    fn check_token_transition(&self, token_text: &str, required_keys: &[String]) -> bool {
        let mut depth = self.depth;
        let mut in_string = self.in_string;
        let mut escape_next = self.escape_next;
        let mut stack = self.delimiter_stack.clone();
        let mut closed_root = false;

        for ch in token_text.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            if in_string {
                if ch == '\\' {
                    escape_next = true;
                } else if ch == '"' {
                    in_string = false;
                }
                continue;
            }
            if closed_root {
                // Once root JSON structure has closed, only trailing whitespace is valid
                if !ch.is_whitespace() {
                    return false;
                }
                continue;
            }
            if ch == '"' {
                in_string = true;
                continue;
            }
            match ch {
                '{' => {
                    depth += 1;
                    stack.push('}');
                }
                '[' => {
                    depth += 1;
                    stack.push(']');
                }
                '}' | ']' => {
                    if depth <= 0 {
                        return false; // Unmatched closing delimiter underflow
                    }
                    if let Some(expected) = stack.pop() {
                        if expected != ch {
                            return false; // Mismatched closing delimiter (e.g. `{]` or `[}`)
                        }
                    }
                    depth -= 1;
                    if depth == 0 {
                        // Prevent closing root object until all required schema keys are present
                        if !required_keys.is_empty() {
                            let combined = format!("{}{}", self.current_buffer, token_text);
                            for req_key in required_keys {
                                let pat = format!("\"{}\"", req_key);
                                if !combined.contains(&pat) {
                                    return false;
                                }
                            }
                        }
                        closed_root = true;
                    }
                }
                _ => {}
            }
        }

        true
    }

    /// Compute token validity mask for vocabulary (<50μs latency budget)
    pub fn compute_validity_mask(&self, vocab_tokens: &[String]) -> Vec<bool> {
        let mut mask = vec![true; vocab_tokens.len()];
        let at_start = self.current_buffer.trim().is_empty();
        let already_terminated = self.is_valid_termination();

        match &self.grammar {
            SchemaGrammar::JsonAny => {
                for (i, tok) in vocab_tokens.iter().enumerate() {
                    let trimmed = tok.trim_start();
                    if at_start {
                        if !trimmed.is_empty()
                            && !trimmed.starts_with('{')
                            && !trimmed.starts_with('[')
                        {
                            mask[i] = false;
                            continue;
                        }
                    } else if already_terminated && !trimmed.is_empty() {
                        mask[i] = false;
                        continue;
                    }
                    if !self.check_token_transition(tok, &[]) {
                        mask[i] = false;
                    }
                }
            }
            SchemaGrammar::JsonObject { required_keys } => {
                for (i, tok) in vocab_tokens.iter().enumerate() {
                    let trimmed = tok.trim_start();
                    if at_start {
                        if !trimmed.is_empty() && !trimmed.starts_with('{') {
                            mask[i] = false;
                            continue;
                        }
                    } else if already_terminated && !trimmed.is_empty() {
                        mask[i] = false;
                        continue;
                    }
                    if !self.check_token_transition(tok, required_keys) {
                        mask[i] = false;
                    }
                }
            }
            SchemaGrammar::Regex(_) => {
                for (i, tok) in vocab_tokens.iter().enumerate() {
                    if !self.check_token_transition(tok, &[]) {
                        mask[i] = false;
                    }
                }
            }
        }

        mask
    }

    pub fn is_valid_termination(&self) -> bool {
        self.depth == 0
            && self.delimiter_stack.is_empty()
            && !self.in_string
            && !self.current_buffer.trim().is_empty()
    }
}

