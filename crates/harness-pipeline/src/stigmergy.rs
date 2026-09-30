use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct PheromoneEdge {
    pub from_state: String,
    pub to_state: String,
    pub pheromone_level: f32, // \tau_{ij}
    pub visit_count: usize,
    pub success_count: usize,
}

/// Ant Colony Stigmergic Trajectory Search & Pruner:
/// Inspired by millions of years of evolutionary optimization in social insect foraging.
/// In autonomous agent tree searches (Tree-of-Thoughts / tool exploration):
/// 1. Pheromone Deposition: Productive tool executions deposit chemical reinforcement (\Delta \tau).
/// 2. Pheromone Evaporation: Dead-end or failing branches naturally evaporate (\tau \leftarrow (1 - \rho) * \tau).
/// 3. Probabilistic Guidance: Future agent trajectories follow pheromone gradients,
///    avoiding repetitive failure loops and pruning stale branches without manual heuristics.
pub struct StigmergicTrajectoryManager {
    pub evaporation_rate: f32, // \rho (typically 0.15 - 0.25)
    pub alpha: f32,            // Pheromone importance exponent
    pub beta: f32,             // Heuristic importance exponent
    edges: HashMap<(String, String), PheromoneEdge>,
}

impl StigmergicTrajectoryManager {
    pub fn new(evaporation_rate: f32, alpha: f32, beta: f32) -> Self {
        Self {
            evaporation_rate,
            alpha,
            beta,
            edges: HashMap::new(),
        }
    }

    /// Record a transition and deposit pheromones on success
    pub fn record_transition(
        &mut self,
        from_state: &str,
        to_state: &str,
        reward: f32, // 1.0 for success, 0.0 for failure
    ) {
        let key = (from_state.to_string(), to_state.to_string());
        let edge = self.edges.entry(key).or_insert(PheromoneEdge {
            from_state: from_state.to_string(),
            to_state: to_state.to_string(),
            pheromone_level: 1.0,
            visit_count: 0,
            success_count: 0,
        });

        edge.visit_count += 1;
        if reward > 0.0 {
            edge.success_count += 1;
            // Pheromone deposition: \tau \leftarrow \tau + Q * reward
            edge.pheromone_level += 2.0 * reward;
        }
    }

    /// Evaporation step (called after each agent turn/iteration)
    /// \tau_{ij} \leftarrow (1 - \rho) * \tau_{ij}
    pub fn evaporate(&mut self) {
        let decay = 1.0 - self.evaporation_rate;
        for edge in self.edges.values_mut() {
            edge.pheromone_level = (edge.pheromone_level * decay).max(0.01);
        }
    }

    /// Calculate probability of choosing to_state from from_state
    pub fn transition_probability(&self, from_state: &str, to_state: &str, heuristic_score: f32) -> f32 {
        let key = (from_state.to_string(), to_state.to_string());
        let tau = self.edges.get(&key).map(|e| e.pheromone_level).unwrap_or(1.0);
        let eta = heuristic_score.max(1e-4);

        tau.powf(self.alpha) * eta.powf(self.beta)
    }

    /// Prune dead-end branches whose pheromone has evaporated below threshold
    pub fn prune_dead_ends(&mut self, threshold: f32) -> usize {
        let initial_len = self.edges.len();
        self.edges.retain(|_, edge| edge.pheromone_level >= threshold);
        initial_len - self.edges.len()
    }
}

impl Default for StigmergicTrajectoryManager {
    fn default() -> Self {
        Self::new(0.20, 1.0, 2.0)
    }
}
