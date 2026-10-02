use harness_pipeline::code_engine::{CodeQualityEngine, Language};
use harness_pipeline::reasoning::{BestOfNConfig, CandidateTrajectory, TestTimeReasoningEngine};

#[test]
fn test_code_quality_json_validation_and_repair() {
    // Valid JSON
    let valid_json = r#"{"model": "llama-3-8b", "temperature": 0.7, "top_p": 0.9}"#;
    let res = CodeQualityEngine::verify_json(valid_json);
    assert!(res.is_valid);
    assert_eq!(res.syntax_score, 1.0);
    assert!(res.errors.is_empty());

    // Truncated/Broken JSON with unclosed braces and quotes
    let broken_json = r#"{"model": "llama-3-8b", "parameters": {"ctx": 4096, "layers": 32"#;
    let res_broken = CodeQualityEngine::verify_json(broken_json);
    assert!(!res_broken.is_valid);
    assert!(res_broken.repaired_code.is_some());

    let repaired = res_broken.repaired_code.unwrap();
    println!("Repaired JSON: {}", repaired);
    // Repaired JSON should now parse validly
    let check = serde_json::from_str::<serde_json::Value>(&repaired);
    assert!(check.is_ok(), "Repaired JSON failed to parse: {:?}", check);
}

#[test]
fn test_code_quality_rust_syntax_checks() {
    let valid_rust = r#"
fn compute_matmul(a: &[f32], b: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0; a.len()];
    for i in 0..a.len() {
        out[i] = a[i] * b[i];
    }
    out
}
"#;
    let res = CodeQualityEngine::verify_rust(valid_rust);
    assert!(res.is_valid);
    assert!(res.errors.is_empty());

    // Broken Rust with unclosed delimiter
    let broken_rust = r#"
fn compute_something(a: i32) -> i32 {
    let x = (a + 5 * (2 + 3);
    x
}
"#;
    let res_broken = CodeQualityEngine::verify_rust(broken_rust);
    assert!(!res_broken.is_valid);
    assert!(!res_broken.unclosed_delimiters.is_empty());
}

#[test]
fn test_code_quality_python_syntax_checks() {
    let valid_py = r#"
def calculate_attention(q, k, v):
    scores = q @ k.T
    return scores @ v
"#;
    let res = CodeQualityEngine::verify_python(valid_py);
    assert!(res.is_valid);

    // Broken Python with missing colon
    let broken_py = r#"
def calculate_attention(q, k, v)
    scores = q @ k.T
"#;
    let res_broken = CodeQualityEngine::verify_python(broken_py);
    assert!(!res_broken.is_valid);
    assert!(res_broken.errors.iter().any(|e| e.contains("trailing colon")));
}

#[test]
fn test_code_quality_markdown_extraction() {
    let markdown = r#"
Here is the solution to your query:

```rust
pub fn solve() -> bool {
    true
}
```

And in Python:
```python
def solve():
    return True
```
"#;
    let blocks = CodeQualityEngine::extract_fenced_blocks(markdown);
    assert_eq!(blocks.len(), 2);
    assert_eq!(blocks[0].0, Language::Rust);
    assert_eq!(blocks[1].0, Language::Python);
}

#[test]
fn test_test_time_reasoning_best_of_n_selection() {
    let config = BestOfNConfig {
        num_candidates: 3,
        max_tokens: 128,
        temperature: 0.7,
        entropy_penalty: 0.5,
        inhibition_strength: 0.4,
        contrast_margin: 1.0,
    };

    let mut engine = TestTimeReasoningEngine::new(config);

    // Candidate 1: Hallucinatory/rambling, high entropy (uncertain)
    let c1 = CandidateTrajectory {
        candidate_id: 1,
        tokens: vec![1, 2, 3],
        text: "Maybe it is X, or could be Y, but perhaps Z...".into(),
        mean_entropy: 3.2, // High uncertainty
        min_contrast_ratio: 0.2,
        quality_score: 0.2 - (0.5 * 3.2), // Negative / low
        generation_time_ms: 12.0,
    };

    // Candidate 2: Crisp, factual, decisive reasoning (low entropy, high contrast)
    let c2 = CandidateTrajectory {
        candidate_id: 2,
        tokens: vec![4, 5, 6],
        text: "Step 1: Compute matrix rank. Step 2: Invert diagonal block. Result is verified 42.".into(),
        mean_entropy: 0.45, // Decisive
        min_contrast_ratio: 2.8,
        quality_score: 2.8 - (0.5 * 0.45), // ~2.575 (Champion)
        generation_time_ms: 10.5,
    };

    // Candidate 3: Moderate
    let c3 = CandidateTrajectory {
        candidate_id: 3,
        tokens: vec![7, 8, 9],
        text: "The result is likely 42 based on standard substitution.".into(),
        mean_entropy: 1.2,
        min_contrast_ratio: 1.4,
        quality_score: 1.4 - (0.5 * 1.2), // ~0.8
        generation_time_ms: 11.0,
    };

    let verdict = engine.select_champion(vec![c1, c2, c3], 33.5).unwrap();

    assert_eq!(verdict.winning_candidate_id, 2);
    assert_eq!(verdict.candidates_evaluated, 3);
    assert!(verdict.champion_text.contains("Result is verified 42"));
    assert!(verdict.champion_score > 2.0);
    println!("Reasoning Verdict Champion: Candidate {} (Score: {:.3}, Entropy: {:.3})", 
        verdict.winning_candidate_id, verdict.champion_score, verdict.champion_mean_entropy);
}
