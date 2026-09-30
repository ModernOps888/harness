use harness_core::{Device, Tensor};
use harness_safety::{
    AgentCircuitBreaker, CircuitBreakerState, ConfidenceScorer, ConstrainedDecoder,
    EntropyDetector, LateralInhibitionFilter, ObservationCompactor, SchemaGrammar,
};

#[test]
fn test_constrained_decoding_json_mask() {
    let decoder = ConstrainedDecoder::new(SchemaGrammar::JsonObject {
        required_keys: vec!["name".into()],
    });

    let vocab = vec![
        "{\"".to_string(),
        "hello".to_string(),
        "123".to_string(),
        " {".to_string(),
    ];

    let mask = decoder.compute_validity_mask(&vocab);
    assert!(mask[0]);  // "{\""
    assert!(!mask[1]); // "hello" is rejected
    assert!(!mask[2]); // "123" is rejected
    assert!(mask[3]);  // " {"
}

#[test]
fn test_shannon_entropy_confidence() {
    let confident_logits = Tensor::from_f32_slice(&[10.0, -5.0, -5.0, -5.0], vec![1, 4], Device::Cpu).unwrap();
    let entropy = EntropyDetector::compute_entropy(&confident_logits).unwrap();

    assert!(entropy < 0.1);

    let scorer = ConfidenceScorer::new(0.8);
    let assessment = scorer.assess(&[entropy], "Test factual text");
    assert!(assessment.is_trustworthy);
    assert!(assessment.overall_confidence_score > 0.9);
}

#[test]
fn test_agent_circuit_breaker_duplicate_detection() {
    let mut breaker = AgentCircuitBreaker::new(3, 2);

    breaker.record_action("bash_execute", "{\"cmd\":\"cat non_existent.txt\"}", true, Some("File not found"));
    let state = breaker.record_action("bash_execute", "{\"cmd\":\"cat non_existent.txt\"}", true, Some("File not found"));

    match state {
        CircuitBreakerState::Tripped { reason, .. } => {
            assert!(reason.contains("Duplicate action loop detected"));
        }
        _ => panic!("Circuit breaker should have tripped on duplicate failing action"),
    }
}

#[test]
fn test_observation_compactor_prunes_bloat() {
    let compactor = ObservationCompactor::new(5, 5, 50);

    let mut big_log = String::new();
    for i in 1..=50 {
        if i == 25 {
            big_log.push_str("FATAL ERROR: connection refused on port 5432\n");
        } else {
            big_log.push_str(&format!("info line {}: processing normal event batch\n", i));
        }
    }

    let result = compactor.compact_tool_output(&big_log, "bash");
    assert!(result.was_truncated);
    assert!(result.compacted_token_estimate < result.original_token_estimate);
    assert!(result.compacted_text.contains("FATAL ERROR"));
}

#[test]
fn test_lateral_inhibition_logit_sharpening() {
    let filter = LateralInhibitionFilter::new(0.5, 1.0);

    let mut logits = vec![10.0, 8.5, 3.0, 2.0, 1.0];
    let initial_contrast = filter.compute_contrast_ratio(&logits);

    filter.sharpen_logits(&mut logits);

    // Winning logit remains 10.0, while the noisy tail has been suppressed
    assert_eq!(logits[0], 10.0);
    assert!(logits[2] < 3.0);
    assert!(logits[3] < 2.0);

    let final_contrast = filter.compute_contrast_ratio(&logits);
    assert!(final_contrast >= initial_contrast);
}
