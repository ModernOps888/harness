use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentChunk {
    pub id: Uuid,
    pub doc_id: String,
    pub chunk_index: usize,
    pub text: String,
    pub token_estimate: usize,
}

pub struct SemanticChunker {
    pub target_chunk_size: usize,
    pub overlap: usize,
}

impl SemanticChunker {
    pub fn new(target_chunk_size: usize, overlap: usize) -> Self {
        Self {
            target_chunk_size,
            overlap,
        }
    }

    /// Split raw text into semantically coherent overlapping chunks
    pub fn chunk_text(&self, doc_id: &str, text: &str) -> Vec<DocumentChunk> {
        let paragraphs: Vec<&str> = text.split("\n\n").collect();
        let mut chunks = Vec::new();
        let mut current_buf = String::new();
        let mut chunk_idx = 0;

        for p in paragraphs {
            if current_buf.len() + p.len() > self.target_chunk_size && !current_buf.is_empty() {
                let token_estimate = current_buf.split_whitespace().count();
                chunks.push(DocumentChunk {
                    id: Uuid::new_v4(),
                    doc_id: doc_id.to_string(),
                    chunk_index: chunk_idx,
                    text: current_buf.clone(),
                    token_estimate,
                });
                chunk_idx += 1;

                // Carry over overlap
                let words: Vec<&str> = current_buf.split_whitespace().collect();
                let overlap_start = words.len().saturating_sub(self.overlap);
                current_buf = words[overlap_start..].join(" ");
                current_buf.push('\n');
            }

            current_buf.push_str(p);
            current_buf.push_str("\n\n");
        }

        if !current_buf.trim().is_empty() {
            let token_estimate = current_buf.split_whitespace().count();
            chunks.push(DocumentChunk {
                id: Uuid::new_v4(),
                doc_id: doc_id.to_string(),
                chunk_index: chunk_idx,
                text: current_buf,
                token_estimate,
            });
        }

        chunks
    }
}
