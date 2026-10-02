use harness_core::gpu::{GpuContext, GpuTensor};

#[test]
fn test_native_gpu_context_and_tensor_roundtrip() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test (no adapter available): {}", e);
            return;
        }
    };

    println!("Targeting GPU: {} ({})", ctx.adapter_name, ctx.backend);

    let original = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let gpu_tensor = GpuTensor::from_f32_slice(&ctx, &original, vec![2, 3]).unwrap();
    let readback = gpu_tensor.to_vec(&ctx).unwrap();

    assert_eq!(original, readback);
}

#[test]
fn test_native_gpu_gemm_accuracy() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test: {}", e);
            return;
        }
    };

    // A: 2x3, B: 3x2
    // A = [[1, 2, 3],
    //      [4, 5, 6]]
    // B = [[7, 8],
    //      [9, 1],
    //      [2, 3]]
    // C = A * B = [[1*7+2*9+3*2, 1*8+2*1+3*3],
    //              [4*7+5*9+6*2, 4*8+5*1+6*3]]
    //   = [[7+18+6, 8+2+9],
    //      [28+45+12, 32+5+18]]
    //   = [[31, 19],
    //      [85, 55]]
    let a_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b_data = vec![7.0f32, 8.0, 9.0, 1.0, 2.0, 3.0];

    let a_gpu = GpuTensor::from_f32_slice(&ctx, &a_data, vec![2, 3]).unwrap();
    let b_gpu = GpuTensor::from_f32_slice(&ctx, &b_data, vec![3, 2]).unwrap();

    let c_gpu = a_gpu.matmul(&ctx, &b_gpu).unwrap();
    let c_res = c_gpu.to_vec(&ctx).unwrap();

    let expected = vec![31.0f32, 19.0, 85.0, 55.0];
    assert_eq!(c_res.len(), expected.len());
    for (actual, exp) in c_res.iter().zip(expected.iter()) {
        assert!((actual - exp).abs() < 1e-4, "Expected {}, got {}", exp, actual);
    }
}

#[test]
fn test_native_gpu_rmsnorm_accuracy() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test: {}", e);
            return;
        }
    };

    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let w_data = vec![1.0f32, 1.0, 1.0, 1.0];

    let x_gpu = GpuTensor::from_f32_slice(&ctx, &x_data, vec![1, 4]).unwrap();
    let w_gpu = GpuTensor::from_f32_slice(&ctx, &w_data, vec![1, 4]).unwrap();

    let y_gpu = x_gpu.rms_norm(&ctx, &w_gpu, 1e-5).unwrap();
    let y_res = y_gpu.to_vec(&ctx).unwrap();

    // CPU reference: mean(1^2 + 2^2 + 3^2 + 4^2) = (1+4+9+16)/4 = 30/4 = 7.5
    // rms = 1 / sqrt(7.5 + 1e-5) = 1 / 2.7386128 = 0.365148
    let mean_sq = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
    let rms = 1.0 / (mean_sq + 1e-5f32).sqrt();
    let expected: Vec<f32> = x_data.iter().map(|&v| v * rms).collect();

    for (actual, exp) in y_res.iter().zip(expected.iter()) {
        assert!((actual - exp).abs() < 1e-4, "Expected {}, got {}", exp, actual);
    }
}

#[test]
fn test_native_gpu_swiglu_accuracy() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test: {}", e);
            return;
        }
    };

    let x_data = vec![0.5f32, -1.0, 2.0, 0.0];
    let g_data = vec![1.0f32, 0.5, -0.5, 2.0];

    let x_gpu = GpuTensor::from_f32_slice(&ctx, &x_data, vec![1, 4]).unwrap();
    let g_gpu = GpuTensor::from_f32_slice(&ctx, &g_data, vec![1, 4]).unwrap();

    let y_gpu = x_gpu.swiglu(&ctx, &g_gpu).unwrap();
    let y_res = y_gpu.to_vec(&ctx).unwrap();

    for i in 0..4 {
        let x = x_data[i];
        let g = g_data[i];
        let sigmoid_x = 1.0 / (1.0 + (-x).exp());
        let expected = (x * sigmoid_x) * g;
        assert!((y_res[i] - expected).abs() < 1e-4, "Index {}: expected {}, got {}", i, expected, y_res[i]);
    }
}

#[test]
fn test_native_gpu_gemv_accuracy() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test: {}", e);
            return;
        }
    };

    // x: [1, 4], W: [4, 3] -> y: [1, 3]
    let x_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let w_data = vec![
        0.5f32, 1.0, -0.5,
        2.0, -1.0, 0.0,
        1.5, 0.5, 2.0,
        -0.5, 1.0, 1.5,
    ];

    let x_gpu = GpuTensor::from_f32_slice(&ctx, &x_data, vec![1, 4]).unwrap();
    let w_gpu = GpuTensor::from_f32_slice(&ctx, &w_data, vec![4, 3]).unwrap();

    let y_gpu = x_gpu.gemv(&ctx, &w_gpu).unwrap();
    let y_res = y_gpu.to_vec(&ctx).unwrap();

    let expected = vec![7.0f32, 4.5, 11.5];
    assert_eq!(y_res.len(), expected.len());
    for (actual, exp) in y_res.iter().zip(expected.iter()) {
        assert!((actual - exp).abs() < 1e-4, "Expected {}, got {}", exp, actual);
    }
}

#[test]
fn test_native_gpu_gemv_q4_accuracy() {
    use harness_core::gpu::GpuQ4Tensor;

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test: {}", e);
            return;
        }
    };

    // K = 64 (2 blocks of 32), N = 2 columns
    let k = 64;
    let n = 2;
    let mut x_data = vec![0.0f32; k];
    for i in 0..k {
        x_data[i] = ((i as f32) * 0.1).sin();
    }

    let mut w_data = vec![0.0f32; k * n];
    for i in 0..(k * n) {
        w_data[i] = ((i as f32) * 0.05).cos();
    }

    let x_gpu = GpuTensor::from_f32_slice(&ctx, &x_data, vec![1, k]).unwrap();
    let w_q4 = GpuQ4Tensor::from_f32_matrix(&ctx, &w_data, k, n).unwrap();

    let y_gpu = x_gpu.gemv_q4(&ctx, &w_q4).unwrap();
    let y_res = y_gpu.to_vec(&ctx).unwrap();

    assert_eq!(y_res.len(), n);
    println!("GPU Q4 GEMV Result: {:?}", y_res);

    // Compute exact CPU dequantized dot product
    let blocks_per_col = (k + 31) / 32;
    // We can re-extract the dequantized weights for each column
    let mut expected_dequant = vec![0.0f32; n];
    for col in 0..n {
        for b in 0..blocks_per_col {
            let start_k = b * 32;
            let end_k = (start_k + 32).min(k);
            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for ki in start_k..end_k {
                let val = w_data[ki * n + col];
                if val < min_val { min_val = val; }
                if val > max_val { max_val = val; }
            }
            let diff = max_val - min_val;
            let scale = if diff > 1e-8 { diff / 15.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            for chunk_idx in 0..4 {
                for nibble_idx in 0..8 {
                    let ki = start_k + chunk_idx * 8 + nibble_idx;
                    if ki < end_k {
                        let val = w_data[ki * n + col];
                        let q = (((val - min_val) * inv_scale).round() as u32).min(15);
                        let dequant_w = (q as f32) * scale + min_val;
                        expected_dequant[col] += x_data[ki] * dequant_w;
                    }
                }
            }
        }
    }

    println!("CPU Dequantized Reference: {:?}", expected_dequant);

    for col in 0..n {
        assert!(
            (y_res[col] - expected_dequant[col]).abs() < 1e-4,
            "Col {}: GPU Q4 {} vs CPU Dequant {}, diff {}",
            col,
            y_res[col],
            expected_dequant[col],
            (y_res[col] - expected_dequant[col]).abs()
        );
    }
}


