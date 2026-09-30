use harness_pipeline::StigmergicTrajectoryManager;

#[test]
fn test_stigmergic_pheromone_evaporation_and_pruning() {
    let mut aco = StigmergicTrajectoryManager::new(0.25, 1.0, 1.0);

    aco.record_transition("Root", "PathA_BashCompile", 1.0);
    aco.record_transition("Root", "PathB_FailingCommand", 0.0);

    let prob_a = aco.transition_probability("Root", "PathA_BashCompile", 1.0);
    let prob_b = aco.transition_probability("Root", "PathB_FailingCommand", 1.0);
    assert!(prob_a > prob_b);

    for _ in 0..5 {
        aco.evaporate();
    }

    let pruned_count = aco.prune_dead_ends(0.35);
    assert_eq!(pruned_count, 1);
}
