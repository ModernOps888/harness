use harness_attention::SpikingAttentionEngine;

#[test]
fn test_lif_spiking_attention_sparsity() {
    let mut lif = SpikingAttentionEngine::new(0.90, 0.45, 0.0);

    // Provide 10 token activation energies, with only token 2 and 7 having high spike energy
    let energies = vec![0.1, 0.15, 0.85, 0.2, 0.1, 0.05, 0.2, 0.95, 0.1, 0.1];
    let spikes = lif.step_spikes(&energies);

    // Only the salient tokens fire spikes
    assert!(spikes[2]);
    assert!(spikes[7]);
    assert!(!spikes[0]);
    assert!(!spikes[4]);

    // Biological sparsity saves >70% of compute FLOPs
    let sparsity = lif.compute_sparsity_ratio(&spikes);
    assert!(sparsity >= 0.70);
}
