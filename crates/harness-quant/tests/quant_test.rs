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
            let x = (i as f32) / (numel as f32) * std::f32::consts::TAU;
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

#[test]
fn test_fp8_e4m3_bit_level_and_group_scales() {
    use harness_quant::{f32_to_fp8_e4m3, fp8_e4m3_to_f32, Fp8Tensor};

    // Exact zero, max normal (448.0), negative normal, and subnormal (2^-9 = 0.001953125)
    assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(0.0)), 0.0);
    assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(448.0)), 448.0);
    assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(-448.0)), -448.0);
    let subnormal = 3.0 / 512.0;
    assert!((fp8_e4m3_to_f32(f32_to_fp8_e4m3(subnormal)) - subnormal).abs() < 1e-6);

    // Heavy-tailed tensor: one 32-element block has a massive outlier (200.0), others are small (~0.1)
    let mut vals = vec![0.1f32; 128];
    vals[0] = 200.0;
    let tensor = Tensor::from_f32_slice(&vals, vec![4, 32], Device::Cpu).unwrap();
    let fp8_grouped = Fp8Tensor::from_f32_tensor(&tensor).unwrap();
    assert_eq!(fp8_grouped.group_scales.len(), 4);

    let rec = fp8_grouped.dequantize().unwrap();
    let rec_slice = rec.as_f32_slice().unwrap();
    // Blocks 1..4 are isolated from the outlier in block 0!
    assert!((rec_slice[64] - 0.1).abs() < 0.01);
}

#[test]
fn test_nf4_grouped_64_and_voronoi_binary_search() {
    use harness_quant::{find_closest_nf4_bin, QuantizedNF4Tensor, NF4_BINS};

    for (idx, &center) in NF4_BINS.iter().enumerate() {
        assert_eq!(find_closest_nf4_bin(center) as usize, idx);
    }

    // 256 elements = 4 groups of 64 with vastly different dynamic scales
    let mut values = Vec::with_capacity(256);
    for g in 0..4 {
        let group_scale = 10.0f32.powi(g - 1); // 0.1, 1.0, 10.0, 100.0
        for i in 0..64 {
            values.push(((i as f32 / 32.0) - 1.0) * group_scale);
        }
    }
    let tensor = Tensor::from_f32_slice(&values, vec![4, 64], Device::Cpu).unwrap();
    let q_nf4 = QuantizedNF4Tensor::from_f32_tensor(&tensor).unwrap();
    assert_eq!(q_nf4.scales.len(), 4);
    assert!(q_nf4.scale > 0.0);

    let deq = q_nf4.dequantize().unwrap();
    let deq_slice = deq.as_f32_slice().unwrap();
    // Group 0 (scale 0.1) preserves high precision despite Group 3 having scale 100.0
    for i in 0..64 {
        assert!((values[i] - deq_slice[i]).abs() < 0.02);
    }
}

#[test]
fn test_q4_imatrix_and_matvec_fused() {
    use harness_core::DType;
    use harness_quant::{ISQEngine, QuantTarget};

    let device = Device::Cpu;
    let rows = 16;
    let cols = 64;
    let values: Vec<f32> = (0..(rows * cols))
        .map(|i| ((i as f32) * 0.073).cos() * 2.0)
        .collect();
    let importance: Vec<f32> = (0..cols)
        .map(|c| if c % 8 == 0 { 50.0 } else { 1.0 })
        .collect();

    let tensor = Tensor::from_f32_slice(&values, vec![rows, cols], device).unwrap();
    let q4 = QuantizedQ4Tensor::from_f32_tensor_imatrix(&tensor, &importance).unwrap();

    let x: Vec<f32> = (0..cols).map(|i| (i as f32) * 0.05 - 1.0).collect();
    let y_fused = q4.matvec_fused(&x).unwrap();
    assert_eq!(y_fused.len(), rows);

    // Compare fused in-register matvec against dequantized reference GEMV
    let deq = q4.dequantize().unwrap();
    let deq_w = deq.as_f32_slice().unwrap();
    for (r, &y_val) in y_fused.iter().enumerate().take(rows) {
        let mut expected = 0.0f32;
        for (c, &x_val) in x.iter().enumerate().take(cols) {
            expected += deq_w[r * cols + c] * x_val;
        }
        assert!(
            (y_val - expected).abs() < 1e-3,
            "Mismatch at row {}: {} vs {}",
            r,
            y_val,
            expected
        );
    }

    // Verify ISQEngine promotes high-salience projections (ffn_down / attn_v)
    let isq = ISQEngine::new(QuantTarget::Balanced);
    let prof = isq.profile_layer("blk.5.ffn_down.weight", &tensor).unwrap();
    assert_eq!(prof.target_dtype, DType::FP8E4M3);
}

#[test]
fn test_q4k_superblock_quantize_dequantize_and_fused_dot() {
    use harness_quant::{
        dequantize_q4_k_bytes, dequantize_q8_0_bytes, f16_to_f32, f32_to_f16, BlockQ4K,
    };

    const QK_K: usize = BlockQ4K::BLOCK_SIZE;

    // Verify IEEE-754 FP16 <-> FP32 roundtrip
    for &val in &[0.0f32, 1.0, -1.0, 0.03125, 12.5, -0.004] {
        let bits = f32_to_f16(val);
        let rec = f16_to_f32(bits);
        assert!((val - rec).abs() < 1e-3, "FP16 roundtrip failed for {}: got {}", val, rec);
    }

    // Generate 256 weights across 8 sub-blocks with varying scales and negative offsets
    let mut weights = [0.0f32; QK_K];
    let mut importance = [1.0f32; QK_K];
    for i in 0..QK_K {
        let sub = (i / 32) as f32;
        weights[i] = ((i as f32 * 0.17).sin() * (0.5 + 0.3 * sub)) - (0.2 * sub);
        if i % 16 == 0 {
            importance[i] = 25.0; // High activation salience channel
        }
    }

    let block_std = BlockQ4K::quantize_from_f32(&weights, None);
    let block_imatrix = BlockQ4K::quantize_from_f32(&weights, Some(&importance));

    let mut deq_std = [0.0f32; QK_K];
    let mut deq_im = [0.0f32; QK_K];
    block_std.dequantize(&mut deq_std);
    block_imatrix.dequantize(&mut deq_im);

    // Check MAE of standard Q4_K super-block
    let mae_std: f32 = weights
        .iter()
        .zip(deq_std.iter())
        .map(|(&a, &b)| (a - b).abs())
        .sum::<f32>()
        / (QK_K as f32);
    assert!(mae_std < 0.12, "Q4_K super-block MAE too high: {}", mae_std);

    // Verify imatrix weighted quantization reduces error on high-importance channels
    let salient_err_std: f32 = (0..QK_K)
        .step_by(16)
        .map(|i| (weights[i] - deq_std[i]).powi(2))
        .sum();
    let salient_err_im: f32 = (0..QK_K)
        .step_by(16)
        .map(|i| (weights[i] - deq_im[i]).powi(2))
        .sum();
    assert!(
        salient_err_im <= salient_err_std + 1e-4,
        "imatrix error {} should be <= std error {}",
        salient_err_im,
        salient_err_std
    );

    // Verify fused in-register dot_f32 matches dequantized dot product exactly
    let mut act = [0.0f32; QK_K];
    for (i, a) in act.iter_mut().enumerate() {
        *a = (i as f32 * 0.09).cos();
    }
    let fused_dot = block_imatrix.dot_f32(&act);
    let ref_dot: f32 = deq_im.iter().zip(act.iter()).map(|(w, x)| w * x).sum();
    assert!(
        (fused_dot - ref_dot).abs() < 1e-3,
        "Fused Q4_K dot {} != ref dot {}",
        fused_dot,
        ref_dot
    );

    // Verify raw byte slice unpacking for GGUF Q4_K (type 12) and Q8_0 (type 7)
    let raw_q4k = block_imatrix.to_bytes();
    let unpacked_q4k = dequantize_q4_k_bytes(&raw_q4k, QK_K);
    assert_eq!(unpacked_q4k.len(), QK_K);
    for i in 0..QK_K {
        assert!((unpacked_q4k[i] - deq_im[i]).abs() < 1e-6);
    }

    let mut raw_q8_0 = vec![0u8; 34];
    let d_f16 = f32_to_f16(0.25);
    raw_q8_0[0..2].copy_from_slice(&d_f16.to_le_bytes());
    for i in 0..32 {
        raw_q8_0[2 + i] = (i as i8 - 16) as u8;
    }
    let unpacked_q8 = dequantize_q8_0_bytes(&raw_q8_0, 32);
    assert_eq!(unpacked_q8[0], -4.0);
    assert_eq!(unpacked_q8[16], 0.0);
    assert_eq!(unpacked_q8[20], 1.0);
}


