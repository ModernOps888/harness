use harness_core::{MoEConfig, ModelConfig};
use harness_pipeline::moe_streamer::MoEOffloadEngine;

#[test]
fn test_moe_budget_allocation_6gb() {
    let mut config = ModelConfig::llama3_70b();
    // Configure as 32-layer MoE with 8 experts per layer, Top-2 active
    config.num_hidden_layers = 32;
    config.hidden_size = 4096;
    config.intermediate_size = 14336;
    config.vocab_size = 128256;
    config.moe = Some(MoEConfig {
        num_routed_experts: 8,
        num_shared_experts: 1,
        num_active_experts: 2,
        routing_top_k: 2,
        norm_topk_prob: true,
    });

    let engine = MoEOffloadEngine::new(6000.0, &config);

    // Verify 6GB VRAM budget allocation breakdown
    assert!(engine.profile.backbone_vram_mb > 1400.0 && engine.profile.backbone_vram_mb < 1600.0);
    assert!(engine.profile.total_vram_mb > 5500.0 && engine.profile.total_vram_mb <= 6000.0);
    assert_eq!(engine.profile.num_experts_per_layer, 8);
    assert_eq!(engine.profile.active_experts_per_token, 2);
    assert!(engine.profile.cached_experts_capacity >= 50);
}

#[test]
fn test_moe_cache_hit_and_pcie_savings() {
    let mut config = ModelConfig::llama3_70b();
    config.num_hidden_layers = 32;
    config.hidden_size = 4096;
    config.intermediate_size = 14336;
    config.vocab_size = 128256;
    config.moe = Some(MoEConfig {
        num_routed_experts: 8,
        num_shared_experts: 1,
        num_active_experts: 2,
        routing_top_k: 2,
        norm_topk_prob: true,
    });

    let mut engine = MoEOffloadEngine::new(6000.0, &config);

    // Simulate 16 layers routing to dynamic secondary experts (fits comfortably in 21 dynamic slots)
    // and 16 layers routing only to pinned expert 0
    let mut layer_selections = Vec::new();
    for l in 0..32 {
        if l < 16 {
            layer_selections.push(vec![0, 1]); // Pinned expert 0 + dynamic expert 1
        } else {
            layer_selections.push(vec![0]);    // Only pinned expert 0
        }
    }

    // First token step: 32 pinned hits + 16 dynamic misses (total 48 requests, 32 hits = 66.7%)
    let metrics_t0 = engine.step_token(0, &layer_selections, 14.0);
    assert!(metrics_t0.cache_hit_rate > 0.60);
    assert!(metrics_t0.pcie_bytes_saved_mb > 2500.0);
    assert!(metrics_t0.projected_tok_s >= 4.0);

    // Second token step with the same working set: all 16 dynamic experts are now cached in the 21 dynamic slots!
    // Total 48 requests, 48 hits = 100% cache hits!
    let metrics_t1 = engine.step_token(1, &layer_selections, 14.0);
    assert_eq!(metrics_t1.cache_hit_rate, 1.0);
    assert_eq!(metrics_t1.pcie_transferred_mb, 0.0);
    // With 0 PCIe transfers, latency is pure GPU compute (~18ms -> >50 tok/s burst)
    assert!(metrics_t1.projected_tok_s > 50.0);
}
