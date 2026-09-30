use harness_pipeline::{
    AdaptiveComparisonReport, AdaptiveSpeculativeConfig, AdaptiveSpeculativeDecoder,
};

#[test]
fn test_adaptive_speculative_lossless_exactness() {
    let config = AdaptiveSpeculativeConfig {
        min_depth: 1,
        max_depth: 8,
        initial_depth: 4,
        low_entropy_thresh: 0.6,
        high_entropy_thresh: 1.8,
        abort_entropy_thresh: 2.4,
    };

    let mut decoder = AdaptiveSpeculativeDecoder::new(config);
    assert_eq!(decoder.current_depth, 4);

    // Scenario 1: Perfect match (low entropy, e.g. 0.2 nats)
    let drafts = vec![101, 102, 103, 104];
    let entropies = vec![0.2, 0.25, 0.3, 0.2];
    let targets = vec![101, 102, 103, 104];

    let result = decoder.verify_step(&drafts, &entropies, &targets);
    assert_eq!(result.accepted_count, 4);
    assert_eq!(result.wasted_tokens, 0);
    assert_eq!(result.accepted_tokens, vec![101, 102, 103, 104]);
    // Depth expands from 4 to 5 because all matched and mean entropy was low
    assert_eq!(result.next_recommended_depth, 5);
    assert_eq!(decoder.current_depth, 5);

    // Scenario 2: Mismatch at index 2 (draft token 303 != target token 999)
    // Emitted tokens must be [201, 202, 999] (target token emitted on mismatch, subsequent drafts pruned)
    let drafts2 = vec![201, 202, 303, 304, 305];
    let entropies2 = vec![0.4, 0.5, 1.9, 2.1, 2.3];
    let targets2 = vec![201, 202, 999, 888, 777];

    let result2 = decoder.verify_step(&drafts2, &entropies2, &targets2);
    assert_eq!(result2.accepted_count, 2);
    assert_eq!(result2.accepted_tokens, vec![201, 202, 999]);
    // 2 accepted + 1 target substitute = 3 tokens accounted for. 2 tokens (304, 305) were discarded
    assert_eq!(result2.wasted_tokens, 2);
    // Depth contracts back down
    assert!(decoder.current_depth < 5);

    // Scenario 3: Entropy spike abort
    let drafts3 = vec![501, 502, 503, 504];
    let entropies3 = vec![0.3, 2.8, 2.9, 3.1]; // Critical spike at token 2
    let targets3 = vec![501, 502, 503, 504];

    let effective_len = decoder.determine_effective_draft_len(&entropies3);
    assert_eq!(effective_len, 1); // Aborts past the high-entropy token

    let _result3 = decoder.verify_step(&drafts3, &entropies3, &targets3);
    // Depth immediately clamped to min_depth (1)
    assert_eq!(decoder.current_depth, 1);
}


#[test]
fn test_adaptive_vs_static_wasted_token_reduction() {
    let config = AdaptiveSpeculativeConfig::default();
    let mut adaptive_decoder = AdaptiveSpeculativeDecoder::new(config);

    // Run 50 steps comparing static depth K=5 vs adaptive K in [1, 8]
    let report: AdaptiveComparisonReport = adaptive_decoder.benchmark_adaptive_vs_static(
        50,
        5,
        15.0, // 15ms target step
        2.5,  // 2.5ms draft step
    );

    assert_eq!(report.num_steps, 50);
    assert!(report.static_tokens_emitted > 0);
    assert!(report.adaptive_tokens_emitted > 0);
    // Adaptive depth must demonstrate wasted token reduction by contracting during uncertain tokens
    assert!(report.wasted_tokens_reduction_pct >= 0.0);
    assert!(report.avg_adaptive_depth >= 1.0 && report.avg_adaptive_depth <= 8.0);
}
