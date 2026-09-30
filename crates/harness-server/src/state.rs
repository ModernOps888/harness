use crate::backend::{BackendConfig, BackendProxy};
use harness_attention::{PagedAttentionManager, RadixPrefixCache};
use harness_core::{DeviceManager, ModelConfig};
use harness_pipeline::{BatchedInferenceEngine, SpeculativeDecoder};
use harness_rag::VectorStore;
use harness_safety::ConfidenceScorer;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;

#[derive(Clone)]
pub struct AppState {
    pub model_name: Arc<RwLock<String>>,
    pub model_config: Arc<RwLock<ModelConfig>>,
    pub device_mgr: Arc<DeviceManager>,
    pub paged_attn: Arc<Mutex<PagedAttentionManager>>,
    pub prefix_cache: Arc<Mutex<RadixPrefixCache>>,
    pub batcher: Arc<Mutex<BatchedInferenceEngine>>,
    pub speculative: Arc<Mutex<SpeculativeDecoder>>,
    pub confidence_scorer: Arc<ConfidenceScorer>,
    pub rag_store: Arc<VectorStore>,
    pub backend: Arc<BackendProxy>,
    pub start_time: Instant,
    pub total_tokens_streamed: Arc<AtomicUsize>,
}

impl AppState {
    pub fn new() -> Self {
        let device_mgr = Arc::new(DeviceManager::detect_primary());
        // 1024 blocks of 16 tokens = 16,384 tokens KV cache pool
        let paged_attn = Arc::new(Mutex::new(PagedAttentionManager::new(16, 1024)));
        let prefix_cache = Arc::new(Mutex::new(RadixPrefixCache::new()));
        let batcher = Arc::new(Mutex::new(BatchedInferenceEngine::new(64)));
        let speculative = Arc::new(Mutex::new(SpeculativeDecoder::new(4)));
        let confidence_scorer = Arc::new(ConfidenceScorer::new(0.85));
        let rag_store = Arc::new(VectorStore::new(4096));
        let backend = Arc::new(BackendProxy::new(BackendConfig::default()));

        Self {
            model_name: Arc::new(RwLock::new("Qwen3.8-27B-ISQ".into())),
            model_config: Arc::new(RwLock::new(ModelConfig::qwen3_27b())),
            device_mgr,
            paged_attn,
            prefix_cache,
            batcher,
            speculative,
            confidence_scorer,
            rag_store,
            backend,
            start_time: Instant::now(),
            total_tokens_streamed: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn with_backend(mut self, config: BackendConfig) -> Self {
        self.backend = Arc::new(BackendProxy::new(config));
        self
    }

    pub fn record_tokens(&self, count: usize) {
        self.total_tokens_streamed.fetch_add(count, Ordering::Relaxed);
    }
}

impl Default for AppState {
    fn default() -> Self {
        Self::new()
    }
}
