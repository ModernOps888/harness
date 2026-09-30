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
