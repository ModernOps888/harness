use harness_rag::HippocampalConsolidator;

#[test]
fn test_hippocampal_engram_consolidation() {
    let mut hippocampus = HippocampalConsolidator::new(100, 64);

    let raw_tokens = 120;
    let dim = 64;
    let simulated_states = vec![0.5f32; raw_tokens * dim];

    assert!(hippocampus.should_consolidate(raw_tokens));

    let engram = hippocampus.consolidate_to_engram("agent_session_1", (0, 120), &simulated_states, dim);
    assert_eq!(engram.dense_representation.len(), dim);

    let savings = hippocampus.memory_savings_ratio(120, 1);
    assert!(savings > 0.98);
}
