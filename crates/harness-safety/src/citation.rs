use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundedClaim {
    pub claim_text: String,
    pub source_id: String,
    pub citation_snippet: String,
    pub similarity_score: f32,
    pub is_supported: bool,
}

pub struct CitationVerifier;

impl CitationVerifier {
    /// Verify generated text claims against retrieved context passages
    pub fn verify_grounding(
        response_text: &str,
        context_passages: &[(&str, &str)], // (source_id, passage_text)
    ) -> Vec<GroundedClaim> {
        let mut claims = Vec::new();

        for sentence in response_text.split(&['.', '\n'][..]) {
            let s = sentence.trim();
            if s.len() < 10 {
                continue;
            }

            let mut best_match = None;
            let mut highest_score = 0.0f32;

            for (src_id, passage) in context_passages {
                // Word overlap / jaccard similarity metric
                let s_words: Vec<&str> = s.split_whitespace().collect();
                let matching = s_words.iter().filter(|w| passage.contains(*w)).count();
                let score = if !s_words.is_empty() {
                    matching as f32 / s_words.len() as f32
                } else {
                    0.0
                };

                if score > highest_score {
                    highest_score = score;
                    best_match = Some((*src_id, *passage));
                }
            }

            if let Some((src_id, passage)) = best_match {
                claims.push(GroundedClaim {
                    claim_text: s.to_string(),
                    source_id: src_id.to_string(),
                    citation_snippet: passage.chars().take(120).collect(),
                    similarity_score: highest_score,
                    is_supported: highest_score >= 0.45,
                });
            }
        }

        claims
    }
}
