use harness_attention::PagedAttentionManager;
use harness_core::{DType, Device, ModelConfig, Tensor};
use harness_pipeline::{BatchedInferenceEngine, GenerationRequest, SamplingConfig, TokenSampler};
use harness_quant::QuantizedQ4Tensor;
use harness_safety::{ConfidenceScorer, ConstrainedDecoder, EntropyDetector, SchemaGrammar};
use uuid::Uuid;

#[test]
fn test_paged_attention_allocation_and_fragmentation() {
    let mut manager = PagedAttentionManager::new(16, 64);
    let req_id = Uuid::new_v4();

    // Allocate 3 blocks
    let b1 = manager.allocate_block(req_id).unwrap();
    let b2 = manager.allocate_block(req_id).unwrap();
    let b3 = manager.allocate_block(req_id).unwrap();

    assert_eq!(manager.get_block_table(&req_id).unwrap().len(), 3);
    assert!(manager.memory_fragmentation_ratio() < 0.05);

    // Free request
    manager.free_request(&req_id);
    assert_eq!(manager.free_block_count(), 64);
}

#[test]
fn test_q4_quantization_roundtrip() {
    let values = vec![0.5f32, -1.2, 3.4, 0.0, -0.8, 2.1, 1.5, -0.3];
    let tensor = Tensor::from_f32_slice(&values, vec![2, 4], Device::Cpu).unwrap();

    let q4 = QuantizedQ4Tensor::from_f32_tensor(&tensor).unwrap();
    let dequant = q4.dequantize().unwrap();

    let dequant_slice = dequant.as_f32_slice().unwrap();
    assert_eq!(dequant_slice.len(), values.len());

    // Verify quantization error is within 4-bit quantization boundary
    for (orig, deq) in values.iter().zip(dequant_slice.iter()) {
        assert!((orig - deq).abs() < 0.5);
    }
}

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
    // At start of JSON object, only tokens starting with '{' are permitted
    assert!(mask[0]);  // "{\""
    assert!(!mask[1]); // "hello" is rejected
    assert!(!mask[2]); // "123" is rejected
    assert!(mask[3]);  // " {"
}

#[test]
fn test_shannon_entropy_confidence() {
    let confident_logits = Tensor::from_f32_slice(&[10.0, -5.0, -5.0, -5.0], vec![1, 4], Device::Cpu).unwrap();
    let entropy = EntropyDetector::compute_entropy(&confident_logits).unwrap();

    // High confidence has very low entropy (< 0.1)
    assert!(entropy < 0.1);

    let scorer = ConfidenceScorer::new(0.8);
    let assessment = scorer.assess(&[entropy], "Test factual text");
    assert!(assessment.is_trustworthy);
    assert!(assessment.overall_confidence_score > 0.9);
}
