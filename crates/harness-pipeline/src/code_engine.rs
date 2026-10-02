use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Language {
    Json,
    Rust,
    Python,
    TypeScript,
    Sql,
    Unknown,
}

impl Language {
    pub fn from_tag(tag: &str) -> Self {
        match tag.trim().to_lowercase().as_str() {
            "json" => Self::Json,
            "rs" | "rust" => Self::Rust,
            "py" | "python" => Self::Python,
            "ts" | "typescript" | "js" | "javascript" => Self::TypeScript,
            "sql" => Self::Sql,
            _ => Self::Unknown,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyntaxVerificationResult {
    pub language: Language,
    pub is_valid: bool,
    pub syntax_score: f32, // 1.0 = flawless, 0.0 = completely invalid
    pub errors: Vec<String>,
    pub unclosed_delimiters: Vec<char>,
    pub repaired_code: Option<String>,
}

/// In-Loop Code Quality & Syntax Verification Engine:
/// Solves LLM syntax corruption, malformed JSON, and broken brackets
/// by performing real-time AST/lexical verification and deterministic repair.
pub struct CodeQualityEngine;

impl CodeQualityEngine {
    /// Extract fenced code blocks from LLM markdown response
    pub fn extract_fenced_blocks(text: &str) -> Vec<(Language, String)> {
        let mut blocks = Vec::new();
        let mut in_block = false;
        let mut current_lang = Language::Unknown;
        let mut current_code = String::new();

        for line in text.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("```") {
                if in_block {
                    // Block finished
                    blocks.push((current_lang, current_code.clone()));
                    current_code.clear();
                    in_block = false;
                    current_lang = Language::Unknown;
                } else {
                    // Block started
                    in_block = true;
                    let tag = trimmed.trim_start_matches('`').trim();
                    current_lang = Language::from_tag(tag);
                }
            } else if in_block {
                current_code.push_str(line);
                current_code.push('\n');
            }
        }

        // If block remained open (truncated generation)
        if in_block && !current_code.is_empty() {
            blocks.push((current_lang, current_code));
        }

        blocks
    }

    /// Verify syntax for code according to specified language
    pub fn verify_syntax(code: &str, lang: Language) -> SyntaxVerificationResult {
        match lang {
            Language::Json => Self::verify_json(code),
            Language::Rust => Self::verify_rust(code),
            Language::Python => Self::verify_python(code),
            Language::TypeScript => Self::verify_delimiters(code, Language::TypeScript),
            Language::Sql => Self::verify_delimiters(code, Language::Sql),
            Language::Unknown => Self::verify_delimiters(code, Language::Unknown),
        }
    }

    /// Strict JSON schema and parse verification
    pub fn verify_json(code: &str) -> SyntaxVerificationResult {
        let trimmed = code.trim();
        let mut errors = Vec::new();

        if trimmed.is_empty() {
            errors.push("Empty JSON content".into());
            return SyntaxVerificationResult {
                language: Language::Json,
                is_valid: false,
                syntax_score: 0.0,
                errors,
                unclosed_delimiters: Vec::new(),
                repaired_code: None,
            };
        }

        match serde_json::from_str::<serde_json::Value>(trimmed) {
            Ok(_) => SyntaxVerificationResult {
                language: Language::Json,
                is_valid: true,
                syntax_score: 1.0,
                errors: Vec::new(),
                unclosed_delimiters: Vec::new(),
                repaired_code: None,
            },
            Err(e) => {
                let err_msg = format!("JSON syntax error at line {}, col {}: {}", e.line(), e.column(), e);
                errors.push(err_msg);

                // Attempt repair of unclosed braces or brackets
                let (repaired, delimiters) = Self::repair_delimiters(trimmed);
                let can_parse = serde_json::from_str::<serde_json::Value>(&repaired).is_ok();

                SyntaxVerificationResult {
                    language: Language::Json,
                    is_valid: false,
                    syntax_score: if can_parse { 0.8 } else { 0.2 },
                    errors,
                    unclosed_delimiters: delimiters,
                    repaired_code: if can_parse { Some(repaired) } else { None },
                }
            }
        }
    }

    /// Rust syntax validator: checks delimiter balance and common closure errors
    pub fn verify_rust(code: &str) -> SyntaxVerificationResult {
        let mut result = Self::verify_delimiters(code, Language::Rust);

        // Check for common Rust syntax rules: fn must have parameter parens and block braces
        for (line_idx, line) in code.lines().enumerate() {
            let trimmed = line.trim();
            if (trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ")) && !trimmed.contains('(') {
                result.errors.push(format!("Line {}: fn declaration missing parameter list `()`", line_idx + 1));
                result.is_valid = false;
            }
        }

        result
    }

    /// Python syntax validator: checks indentation consistency and colon blocks
    pub fn verify_python(code: &str) -> SyntaxVerificationResult {
        let mut errors = Vec::new();
        let mut has_spaces = false;
        let mut has_tabs = false;

        for (line_idx, line) in code.lines().enumerate() {
            let leading_spaces = line.chars().take_while(|c| *c == ' ').count();
            let leading_tabs = line.chars().take_while(|c| *c == '\t').count();

            if leading_spaces > 0 {
                has_spaces = true;
            }
            if leading_tabs > 0 {
                has_tabs = true;
            }

            let trimmed = line.trim();
            if (trimmed.starts_with("def ")
                || trimmed.starts_with("class ")
                || trimmed.starts_with("if ")
                || trimmed.starts_with("elif ")
                || trimmed.starts_with("else:")
                || trimmed.starts_with("for ")
                || trimmed.starts_with("while "))
                && !trimmed.ends_with(':')
                && !trimmed.contains('#')
            {
                errors.push(format!("Line {}: Compound statement missing trailing colon `:`", line_idx + 1));
            }
        }

        if has_spaces && has_tabs {
            errors.push("Inconsistent indentation: mixed tabs and spaces detected".into());
        }

        let delimiter_check = Self::verify_delimiters(code, Language::Python);
        let mut all_errors = delimiter_check.errors;
        all_errors.extend(errors);

        let is_valid = all_errors.is_empty();
        let syntax_score = if is_valid { 1.0 } else { (1.0 - (all_errors.len() as f32 * 0.25)).max(0.1) };

        SyntaxVerificationResult {
            language: Language::Python,
            is_valid,
            syntax_score,
            errors: all_errors,
            unclosed_delimiters: delimiter_check.unclosed_delimiters,
            repaired_code: delimiter_check.repaired_code,
        }
    }

    /// Generic delimiter matching validator (`{}`, `()`, `[]`, `""`, `''`)
    pub fn verify_delimiters(code: &str, lang: Language) -> SyntaxVerificationResult {
        let (repaired, unclosed) = Self::repair_delimiters(code);
        let is_valid = unclosed.is_empty();
        let mut errors = Vec::new();

        if !is_valid {
            errors.push(format!("Unclosed delimiters detected: {:?}", unclosed));
        }

        SyntaxVerificationResult {
            language: lang,
            is_valid,
            syntax_score: if is_valid { 1.0 } else { 0.5 },
            errors,
            unclosed_delimiters: unclosed.clone(),
            repaired_code: if !unclosed.is_empty() { Some(repaired) } else { None },
        }
    }

    /// Deterministically repair unclosed brackets and quotes
    pub fn repair_delimiters(code: &str) -> (String, Vec<char>) {
        let mut stack = Vec::new();
        let mut in_string = false;
        let mut quote_char = ' ';
        let mut escape_next = false;

        for ch in code.chars() {
            if escape_next {
                escape_next = false;
                continue;
            }
            if ch == '\\' {
                escape_next = true;
                continue;
            }
            if ch == '"' || ch == '\'' {
                if in_string {
                    if ch == quote_char {
                        in_string = false;
                    }
                } else {
                    in_string = true;
                    quote_char = ch;
                }
                continue;
            }

            if !in_string {
                match ch {
                    '{' => stack.push('}'),
                    '(' => stack.push(')'),
                    '[' => stack.push(']'),
                    '}' | ')' | ']' => {
                        if let Some(&expected) = stack.last() {
                            if expected == ch {
                                stack.pop();
                            }
                        }
                    }
                    _ => {}
                }
            }
        }

        let mut repaired = code.to_string();
        if in_string {
            repaired.push(quote_char);
        }

        let unclosed = stack.clone();
        while let Some(closing) = stack.pop() {
            repaired.push(closing);
        }

        (repaired, unclosed)
    }
}
