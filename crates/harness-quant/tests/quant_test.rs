use harness_core::{Device, Tensor};
use harness_quant::{
    dequantize_fp8, dequantize_nf4, quantize_fp8, quantize_nf4, QuantizedQ4Tensor,
};

#[test]
fn test_nf4_quantization_snr_and_compression() {
    let device = Device::Cpu;
    let shape = vec![64, 128]; // 8192 elements = 32768 bytes in FP32
    let numel = 64 * 128;

    // Generate normal-like values in [-2.5, 2.5]
    let values: Vec<f32> = (0..numel)
        .map(|i| {
            let x = (i as f32) / (numel as f32) * 6.28318;
            (x.sin() + 0.5 * (x * 2.0).cos()) * 1.5
        })
        .collect();

    let orig_tensor = Tensor::from_f32_slice(&values, shape.clone(), device.clone()).unwrap();

    // Quantize to NF4
    let (packed, scale) = quantize_nf4(&orig_tensor).unwrap();
    // 8192 elements packed 2 per byte = 4096 bytes
    assert_eq!(packed.len(), 4096);
    assert!(scale > 0.0);

    // Dequantize back to FP32
    let dequant_tensor = dequantize_nf4(&packed, scale, shape.clone(), device).unwrap();
    let dequant_slice = dequant_tensor.as_f32_slice().unwrap();

    // Verify Mean Absolute Error is bounded
    let mut sum_abs_err = 0.0f32;
    let mut max_err = 0.0f32;
    for i in 0..numel {
        let err = (values[i] - dequant_slice[i]).abs();
        sum_abs_err += err;
        if err > max_err {
            max_err = err;
        }
    }
    let mae = sum_abs_err / numel as f32;

    // NF4 4-bit non-linear quantization with 16 bins achieves MAE < 0.12 on dynamic waves
    assert!(mae < 0.12, "NF4 MAE too high: {}", mae);
}

#[test]
fn test_q4_block_quantization_roundtrip() {
    let device = Device::Cpu;
    let numel = 1024;
    let shape = vec![32, 32];

    let values: Vec<f32> = (0..numel)
        .map(|i| (i as f32 * 0.137).sin() * 10.0)
        .collect();

    let tensor = Tensor::from_f32_slice(&values, shape.clone(), device).unwrap();
    let q4 = QuantizedQ4Tensor::from_f32_tensor(&tensor).unwrap();

    // 1024 elements / 32 per block = 32 blocks
    assert_eq!(q4.blocks.len(), 32);

    // Dequantize back
    let reconstructed = q4.dequantize().unwrap();
    let rec_slice = reconstructed.as_f32_slice().unwrap();

    for i in 0..numel {
        let orig = values[i];
        let rec = rec_slice[i];
        // 4-bit linear quantization gives 16 levels: error bounded by block_range / 15
        assert!((orig - rec).abs() < 1.5, "Diff at {}: orig={}, rec={}", i, orig, rec);
    }
}

#[test]
fn test_fp8_quantization_roundtrip() {
    let device = Device::Cpu;
    let numel = 512;
    let shape = vec![16, 32];

    let values: Vec<f32> = (0..numel)
        .map(|i| (i as f32 - 256.0) * 0.05)
        .collect();

    let tensor = Tensor::from_f32_slice(&values, shape.clone(), device.clone()).unwrap();
    let (packed, inv_scale) = quantize_fp8(&tensor).unwrap();
    assert_eq!(packed.len(), 512); // 1 byte per element

    let reconstructed = dequantize_fp8(&packed, inv_scale, shape, device).unwrap();
    let rec_slice = reconstructed.as_f32_slice().unwrap();

    let mut sum_err = 0.0f32;
    for i in 0..numel {
        sum_err += (values[i] - rec_slice[i]).abs();
    }
    let mae = sum_err / numel as f32;
    assert!(mae < 0.2, "FP8 MAE too high: {}", mae);
}
