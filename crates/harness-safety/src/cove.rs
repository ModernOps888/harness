use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationStep {
    pub question: String,
    pub independent_answer: String,
    pub is_consistent: bool,
}

/// Chain-of-Verification (CoVe) engine:
/// 1. Draft baseline response
/// 2. Plan factual verification questions
/// 3. Answer questions independently to avoid hallucination bias
/// 4. Generate final verified response correcting any discrepancy
pub struct ChainOfVerification;

impl ChainOfVerification {
    pub fn plan_verification_queries(draft: &str) -> Vec<String> {
        let mut questions = Vec::new();
        // Extract factual assertions and generate fact-checking questions
        for sentence in draft.split(&['.', ';', '\n'][..]) {
            let s = sentence.trim();
            if s.len() > 15 && (s.contains("is") || s.contains("was") || s.contains("has") || s.contains("released")) {
                questions.push(format!("Is it verified that {}?", s));
            }
        }
        questions.truncate(3); // Cap verification queries to limit token consumption
        questions
    }
}
