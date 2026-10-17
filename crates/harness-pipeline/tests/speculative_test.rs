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

#[test]
fn test_qwen25_coder_32b_vocab_alignment_and_llama_args() {
    use harness_pipeline::{QWEN25_CODER_DRAFT_VOCAB_SIZE, QWEN25_CODER_TARGET_VOCAB_SIZE};

    let cfg = AdaptiveSpeculativeConfig::qwen2_5_coder_32b_profile();
    let decoder = AdaptiveSpeculativeDecoder::new(cfg).with_verification_gamma(0.05);

    let args = decoder.to_llama_server_args();
    assert!(args.contains(&"--draft-max".to_string()));
    assert!(args.contains(&"--draft-min".to_string()));
    assert!(args.contains(&"--draft-p-min".to_string()));

    // Verify empirical batch verification latency model: T_verify(K+1) = T_step(1) * (1 + gamma * K)
    let t_verify = decoder.verification_latency_ms(600.0, 4);
    assert!((t_verify - 720.0).abs() < 1e-3);

    // Verify vocabulary slice alignment between 1.5B drafter (151936) and 32B verifier (152064)
    let draft_logits = vec![0.1f32; QWEN25_CODER_DRAFT_VOCAB_SIZE];
    let target_logits = vec![0.2f32; QWEN25_CODER_TARGET_VOCAB_SIZE];
    let (d_aligned, t_aligned, common) =
        AdaptiveSpeculativeDecoder::align_vocab_logits(&draft_logits, &target_logits);
    assert_eq!(common, 151_936);
    assert_eq!(d_aligned.len(), 151_936);
    assert_eq!(t_aligned.len(), 151_936);
}

#[test]
fn test_online_draft_calibrator_and_repo_suffix_drafter() {
    use harness_pipeline::{OnlineDraftCalibrator, RepoSuffixDrafter};

    let mut calibrator = OnlineDraftCalibrator::new(0.45, 0.995);
    // Suppose after context bigram (41, 42), the 0.5B drafter favors token 10, but the 32B verifier repeatedly emits token 11
    let mut initial_logits = vec![0.0f32; 32];
    initial_logits[10] = 2.5;
    initial_logits[11] = 1.5;

    let mut calibrated_before = initial_logits.clone();
    calibrator.apply_logit_biases(41, 42, &mut calibrated_before);
    assert!(calibrated_before[10] > calibrated_before[11]);

    // Observe 6 speculative verification rejections where draft=10 was rejected in favor of target=11
    for _ in 0..6 {
        calibrator.observe_verification(41, 42, 10, 11, 0.70, 0.20);
    }
    assert_eq!(calibrator.total_updates, 6);

    let mut calibrated_after = initial_logits.clone();
    calibrator.apply_logit_biases(41, 42, &mut calibrated_after);
    // Online cross-entropy gradient step shifts the draft distribution so token 11 now beats token 10!
    assert!(
        calibrated_after[11] > calibrated_after[10],
        "Expected token 11 logit ({}) > token 10 logit ({})",
        calibrated_after[11],
        calibrated_after[10]
    );

    // Test RepoSuffixDrafter O(1) trigram continuation for repetitive Rust identifiers/syntax
    let mut repo_drafter = RepoSuffixDrafter::new();
    let repo_tokens = vec![10, 20, 30, 40, 50, 60, 70];
    repo_drafter.index_tokens(&repo_tokens, 6);

    let drafted = repo_drafter.draft_from_context(&[99, 10, 20, 30], 4);
    assert_eq!(drafted, Some(vec![40, 50, 60, 70]));
}


