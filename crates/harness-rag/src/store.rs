use crate::chunker::DocumentChunk;
use harness_core::{Result, Tensor};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::sync::RwLock;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScoredDocument {
    pub chunk: DocumentChunk,
    pub similarity_score: f32,
}

pub struct VectorStore {
    chunks: RwLock<Vec<DocumentChunk>>,
    embeddings: RwLock<Vec<Vec<f32>>>,
    dim: usize,
}

impl VectorStore {
    pub fn new(dim: usize) -> Self {
        Self {
            chunks: RwLock::new(Vec::new()),
            embeddings: RwLock::new(Vec::new()),
            dim,
        }
    }

    pub fn insert(&self, chunk: DocumentChunk, embedding: Vec<f32>) {
        let mut chunks = self.chunks.write().unwrap();
        let mut embeddings = self.embeddings.write().unwrap();
        chunks.push(chunk);
        embeddings.push(embedding);
    }

    /// Fast parallel cosine similarity search
    pub fn search(&self, query_emb: &[f32], top_k: usize) -> Vec<ScoredDocument> {
        let chunks = self.chunks.read().unwrap();
        let embeddings = self.embeddings.read().unwrap();

        if embeddings.is_empty() {
            return Vec::new();
        }

        let query_norm = query_emb.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);

        let mut scored: Vec<(usize, f32)> = embeddings
            .par_iter()
            .enumerate()
            .map(|(i, emb)| {
                let dot: f32 = emb.iter().zip(query_emb.iter()).map(|(&a, &b)| a * b).sum();
                let doc_norm = emb.iter().map(|&x| x * x).sum::<f32>().sqrt().max(1e-8);
                let cos_sim = dot / (query_norm * doc_norm);
                (i, cos_sim)
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(idx, score)| ScoredDocument {
                chunk: chunks[idx].clone(),
                similarity_score: score,
            })
            .collect()
    }
}
