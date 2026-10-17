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

    // Multi-block proportional split: 4 tokens across 2 blocks (2 tokens/block)
    let mut cache2 = RadixPrefixCache::new();
    cache2.insert(&[10, 20, 30, 40], &[101, 102]);
    cache2.insert(&[10, 20, 50, 60], &[101, 202]);

    // Shared 2-token prefix [10, 20] retains physical block [101]!
    let (p_len, p_blocks) = cache2.match_prefix(&[10, 20, 99]);
    assert_eq!(p_len, 2);
    assert_eq!(p_blocks, vec![101]);

    let (b1_len, b1_blocks) = cache2.match_prefix(&[10, 20, 30, 40]);
    assert_eq!(b1_len, 4);
    assert_eq!(b1_blocks, vec![101, 102]);

    let (b2_len, b2_blocks) = cache2.match_prefix(&[10, 20, 50, 60]);
    assert_eq!(b2_len, 4);
    assert_eq!(b2_blocks, vec![101, 202]);
}

#[test]
fn test_kv_cache_fp8_and_fp4_quantized_append_and_readback() {
    use harness_attention::{KVCache, KVCacheCompression};
    use harness_core::Tensor;

    let k_data: Vec<f32> = (0..64).map(|v| (v as f32 - 32.0) * 0.25).collect();
    let v_data: Vec<f32> = (0..64).map(|v| (v as f32 - 16.0) * 0.10).collect();
    let k_new = Tensor::from_f32_slice(&k_data, vec![2, 2, 16], Device::Cpu).unwrap();
    let v_new = Tensor::from_f32_slice(&v_data, vec![2, 2, 16], Device::Cpu).unwrap();

    // FP8E4M3 KV Cache
    let mut cache_fp8 =
        KVCache::new(1, 2, 16, 8, Device::Cpu, KVCacheCompression::FP8E4M3).unwrap();
    cache_fp8.append(&k_new, &v_new).unwrap();
    assert_eq!(cache_fp8.current_len, 2);
    assert_eq!(cache_fp8.memory_bytes(), 2 * 8 * 2 * 16); // 1 byte/elem = 512 bytes

    let (k_act8, v_act8) = cache_fp8.get_active_kv().unwrap();
    assert_eq!(k_act8.shape(), &[2, 2, 16]);
    let k8_slice = k_act8.as_f32_slice().unwrap();
    let v8_slice = v_act8.as_f32_slice().unwrap();
    for i in 0..64 {
        assert!((k8_slice[i] - k_data[i]).abs() < 0.6);
        assert!((v8_slice[i] - v_data[i]).abs() < 0.3);
    }

    // FP4 KV Cache
    let mut cache_fp4 = KVCache::new(1, 2, 16, 8, Device::Cpu, KVCacheCompression::FP4).unwrap();
    cache_fp4.append(&k_new, &v_new).unwrap();
    assert_eq!(cache_fp4.current_len, 2);
    assert_eq!(cache_fp4.memory_bytes(), 2 * (8 * 2 * 16) / 2); // 0.5 byte/elem = 256 bytes

    let (k_act4, v_act4) = cache_fp4.get_active_kv().unwrap();
    assert_eq!(k_act4.shape(), &[2, 2, 16]);
    let k4_slice = k_act4.as_f32_slice().unwrap();
    let v4_slice = v_act4.as_f32_slice().unwrap();
    for i in 0..64 {
        assert!((k4_slice[i] - k_data[i]).abs() < 1.2);
        assert!((v4_slice[i] - v_data[i]).abs() < 0.6);
    }
}

#[test]
fn test_flash_attention_v3_gqa_and_decode_causal_mask() {
    use harness_attention::{flash_attention_v3, FlashAttentionConfig};
    use harness_core::Tensor;

    // Qwen2.5-Coder-32B GQA ratio 5:1 (e.g., 5 query heads, 1 KV head, head_dim = 4)
    // Decode step: seq_len_q = 1, seq_len_kv = 4
    let config = FlashAttentionConfig::new(5, 1, 4);
    let q = Tensor::from_f32_slice(&[1.0f32; 5 * 4], vec![1, 5, 4], Device::Cpu).unwrap();

    // Token 0..2 have V = 0.0, Token 3 (the latest token in KV cache) has huge K and V = 10.0
    let mut k_data = vec![0.0f32; 4 * 4];
    let mut v_data = vec![0.0f32; 4 * 4];
    for d in 0..4 {
        k_data[3 * 4 + d] = 10.0;
        v_data[3 * 4 + d] = 10.0;
    }
    let k = Tensor::from_f32_slice(&k_data, vec![4, 1, 4], Device::Cpu).unwrap();
    let v = Tensor::from_f32_slice(&v_data, vec![4, 1, 4], Device::Cpu).unwrap();

    let out = flash_attention_v3(&q, &k, &v, &config).unwrap();
    assert_eq!(out.shape(), &[1, 5, 4]);
    let out_slice = out.as_f32_slice().unwrap();
    // Because decode query at seq_len_q=1 attends across all 4 KV tokens [0..=3],
    // token 3 dominates the softmax and output must be ~10.0 (not 0.0!)
    assert!(out_slice[0] > 9.9, "Causal decode mask failed to attend to tail KV tokens: {}", out_slice[0]);
}

#[test]
fn test_spiking_attention_soft_reset_no_key_amnesia() {
    use harness_attention::SpikingAttentionEngine;

    let mut engine = SpikingAttentionEngine::new(0.90, 0.50, 0.0);
    // High-salience syntax token receives strong synaptic current across two consecutive decode steps
    let step1 = engine.step_spikes(&[0.60, 0.10]);
    assert_eq!(step1, vec![true, false]);

    // With soft threshold subtraction, remaining potential is 0.60 - 0.25 = 0.35,
    // so a moderate current (0.25) on step 2 still fires (0.90 * 0.35 + 0.25 = 0.565 >= 0.50)!
    let step2 = engine.step_spikes(&[0.25, 0.10]);
    assert_eq!(step2, vec![true, false]);
}

