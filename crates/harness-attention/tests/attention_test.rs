use harness_attention::{
    AttentionSinkManager, ChunkedPrefillEngine, MultiHeadLatentAttention,
};
use harness_core::Device;

#[test]
fn test_attention_sink_manager_rolling_window() {
    let sink = AttentionSinkManager::new(4, 100);

    // Short context (< 104 tokens): retains all tokens
    let indices_short = sink.active_token_indices(50);
    assert_eq!(indices_short.len(), 50);

    // Long context (1000 tokens): retains exactly 4 sink tokens + 100 recent tokens = 104 tokens
    let indices_long = sink.active_token_indices(1000);
    assert_eq!(indices_long.len(), 104);
    assert_eq!(indices_long[0], 0);
    assert_eq!(indices_long[1], 1);
    assert_eq!(indices_long[2], 2);
    assert_eq!(indices_long[3], 3);
    assert_eq!(indices_long[4], 900);
    assert_eq!(indices_long[103], 999);

    assert!(sink.vram_savings_ratio(1000) > 0.89);
}

#[test]
fn test_chunked_prefill_planning() {
    let prefill_engine = ChunkedPrefillEngine::new(512);

    let chunks = prefill_engine.plan_chunks(2048);
    assert_eq!(chunks.len(), 4);
    assert_eq!(chunks[0], (0, 512));
    assert_eq!(chunks[1], (512, 1024));
    assert_eq!(chunks[2], (1024, 1536));
    assert_eq!(chunks[3], (1536, 2048));

    assert!(prefill_engine.estimated_decode_jitter_reduction(4096) > 0.8);
}

#[test]
fn test_multi_head_latent_attention_compression() {
    let mla = MultiHeadLatentAttention::new(4096, 512, 32, 128, &Device::Cpu).unwrap();
    // 32 heads * 128 head_dim * 2 (K and V) = 8,192 scalars
    // MLA caches only 512 latent dimensions -> 93.75% reduction!
    let ratio = mla.compression_ratio();
    assert!(ratio > 0.90);
}

#[test]
fn test_radix_prefix_cache_reuse() {
    use harness_attention::RadixPrefixCache;
    let mut cache = RadixPrefixCache::new();

    // Cache a system prompt token sequence with physical block IDs [101, 102]
    let sys_tokens = vec![1, 1500, 2048, 99];
    let sys_blocks = vec![101, 102];
    cache.insert(&sys_tokens, &sys_blocks);

    assert_eq!(cache.total_cached_tokens(), 4);

    // Query 1: Matching prefix
    let query_tokens = vec![1, 1500, 2048, 99, 4200, 888];
    let (matched_len, reusable_blocks) = cache.match_prefix(&query_tokens);
    assert_eq!(matched_len, 4);
    assert_eq!(reusable_blocks, vec![101, 102]);

    // Query 2: Non-matching prefix
    let non_matching = vec![2, 1500, 2048];
    let (no_match_len, no_blocks) = cache.match_prefix(&non_matching);
    assert_eq!(no_match_len, 0);
    assert!(no_blocks.is_empty());
}

#[test]
fn test_paged_attention_exact_fragmentation_and_churn() {
    use harness_attention::PagedAttentionManager;
    use uuid::Uuid;

    let mut manager = PagedAttentionManager::new(16, 100);
    assert_eq!(manager.total_block_count(), 100);
    assert_eq!(manager.free_block_count(), 100);
    assert_eq!(manager.memory_fragmentation_ratio(), 0.0);

    let req1 = Uuid::new_v4();
    let _b1 = manager.allocate_block(req1).unwrap();
    let _b2 = manager.allocate_block(req1).unwrap();
    assert_eq!(manager.free_block_count(), 98);

    // 2 blocks of 16 tokens = 32 capacity. If req1 generates 32 tokens, 0% fragmentation!
    manager.record_tokens(&req1, 32);
    assert_eq!(manager.memory_fragmentation_ratio(), 0.0);

    // If req2 only uses 24 tokens (8 unused in tail block), fragmentation is exactly 8 / 32 = 0.25 (25%)
    let mut manager2 = PagedAttentionManager::new(16, 100);
    let req2 = Uuid::new_v4();
    let _ = manager2.allocate_block(req2).unwrap();
    let _ = manager2.allocate_block(req2).unwrap();
    manager2.record_tokens(&req2, 24);
    assert!((manager2.memory_fragmentation_ratio() - 0.25).abs() < 1e-5);

    // Free request and verify complete leak-free reclamation
    manager2.free_request(&req2);
    assert_eq!(manager2.free_block_count(), 100);
    assert_eq!(manager2.memory_fragmentation_ratio(), 0.0);
}

#[test]
fn test_kv_cache_append_and_memory_persistence() {
    use harness_attention::{KVCache, KVCacheCompression};
    use harness_core::{Device, Tensor};

    let mut cache = KVCache::new(1, 2, 4, 16, Device::Cpu, KVCacheCompression::None).unwrap();
    assert_eq!(cache.current_len, 0);

    // Create new K and V tokens for 2 tokens (2 tokens * 2 heads * 4 dim = 16 floats each)
    let k_data: Vec<f32> = (1..=16).map(|v| v as f32).collect();
    let v_data: Vec<f32> = (17..=32).map(|v| v as f32).collect();

    let k_new = Tensor::from_f32_slice(&k_data, vec![2, 2, 4], Device::Cpu).unwrap();
    let v_new = Tensor::from_f32_slice(&v_data, vec![2, 2, 4], Device::Cpu).unwrap();

    cache.append(&k_new, &v_new).unwrap();
    assert_eq!(cache.current_len, 2);

    let k_slice = cache.k.as_f32_slice().unwrap();
    let v_slice = cache.v.as_f32_slice().unwrap();

    assert_eq!(&k_slice[0..16], &k_data[..]);
    assert_eq!(&v_slice[0..16], &v_data[..]);
    // The rest of the buffer remains 0
    assert_eq!(k_slice[16], 0.0);
    assert_eq!(v_slice[16], 0.0);
}

#[test]
fn test_radix_prefix_cache_branch_splitting() {
    use harness_attention::RadixPrefixCache;
    let mut cache = RadixPrefixCache::new();

    // Insert prefix 1: [1, 2, 3] -> [10]
    cache.insert(&[1, 2, 3], &[10]);

    // Insert prefix 2: [1, 2, 4] -> [20] (shares [1, 2])
    cache.insert(&[1, 2, 4], &[20]);

    // Query 1: should match [1, 2, 3]
    let (match1_len, blocks1) = cache.match_prefix(&[1, 2, 3, 99]);
    assert_eq!(match1_len, 3);
    assert_eq!(blocks1, vec![10]);

    // Query 2: should match [1, 2, 4]
    let (match2_len, blocks2) = cache.match_prefix(&[1, 2, 4, 100]);
    assert_eq!(match2_len, 3);
    assert_eq!(blocks2, vec![20]);

    // Query 3: partial match [1, 2, 5] should match prefix of length 2 ([1, 2])
    let (match3_len, _blocks3) = cache.match_prefix(&[1, 2, 5]);
    assert_eq!(match3_len, 2);
}
