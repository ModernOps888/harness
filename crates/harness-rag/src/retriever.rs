use crate::store::{ScoredDocument, VectorStore};
use std::sync::Arc;

pub struct RAGRetriever {
    store: Arc<VectorStore>,
}

impl RAGRetriever {
    pub fn new(store: Arc<VectorStore>) -> Self {
        Self { store }
    }

    /// Retrieve top-k context passages and format into a grounded prompt context
    pub fn retrieve_context(&self, query_emb: &[f32], top_k: usize) -> (String, Vec<ScoredDocument>) {
        let results = self.store.search(query_emb, top_k);
        let mut context_str = String::new();

        context_str.push_str("=== GROUNDED CONTEXT SOURCES ===\n");
        for (idx, doc) in results.iter().enumerate() {
            context_str.push_str(&format!(
                "[Source {} - doc_id: {} (Score: {:.2})]\n{}\n\n",
                idx + 1,
                doc.chunk.doc_id,
                doc.similarity_score,
                doc.chunk.text.trim()
            ));
        }
        context_str.push_str("================================\n");

        (context_str, results)
    }
}
