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

    let expected = [31.0f32, 19.0, 85.0, 55.0];
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

    let expected = [7.0f32, 4.5, 11.5];
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
    let x_data: Vec<f32> = (0..k).map(|i| ((i as f32) * 0.1).sin()).collect();
    let w_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.05).cos()).collect();

    let x_gpu = GpuTensor::from_f32_slice(&ctx, &x_data, vec![1, k]).unwrap();
    let w_q4 = GpuQ4Tensor::from_f32_matrix(&ctx, &w_data, k, n).unwrap();

    let y_gpu = x_gpu.gemv_q4(&ctx, &w_q4).unwrap();
    let y_res = y_gpu.to_vec(&ctx).unwrap();

    assert_eq!(y_res.len(), n);
    println!("GPU Q4 GEMV Result: {:?}", y_res);

    // Compute exact CPU dequantized dot product
    let blocks_per_col = k.div_ceil(32);
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

#[test]
fn test_native_gpu_rope_neox_half_split_accuracy() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test: {}", e);
            return;
        }
    };

    let num_tokens = 2;
    let num_heads = 2;
    let head_dim = 8;
    let half_dim = head_dim / 2;
    let start_pos = 3;
    let freq_base = 1_000_000.0f32;

    let total_elems = num_tokens * num_heads * head_dim;
    let x_data: Vec<f32> = (0..total_elems).map(|i| (i as f32) * 0.25 + 0.5).collect();

    let x_gpu = GpuTensor::from_f32_slice(&ctx, &x_data, vec![num_tokens, num_heads, head_dim]).unwrap();
    let y_gpu = x_gpu.rope(&ctx, num_heads, head_dim, start_pos, freq_base).unwrap();
    let y_res = y_gpu.to_vec(&ctx).unwrap();

    let mut expected = x_data.clone();
    for t in 0..num_tokens {
        let pos = (start_pos + t) as f32;
        for h in 0..num_heads {
            let base = (t * num_heads + h) * head_dim;
            for i in 0..half_dim {
                let exponent = (2 * i) as f32 / (head_dim as f32);
                let freq = 1.0 / freq_base.powf(exponent);
                let theta = pos * freq;
                let cos_t = theta.cos();
                let sin_t = theta.sin();

                let x0 = x_data[base + i];
                let x1 = x_data[base + i + half_dim];
                expected[base + i] = x0 * cos_t - x1 * sin_t;
                expected[base + i + half_dim] = x0 * sin_t + x1 * cos_t;
            }
        }
    }

    for i in 0..total_elems {
        assert!(
            (y_res[i] - expected[i]).abs() < 1e-4,
            "RoPE NeoX mismatch at {}: expected {}, got {}",
            i,
            expected[i],
            y_res[i]
        );
    }
}

#[test]
fn test_native_gpu_gemm_q4_batched_speculative_m5() {
    use harness_core::gpu::GpuQ4Tensor;

    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test: {}", e);
            return;
        }
    };

    // Speculative verification batch: M = K_draft + 1 = 5 tokens verified simultaneously!
    // K = 256 (8 blocks of 32), N = 4 output columns
    let m = 5;
    let k = 256;
    let n = 4;

    let x_batch: Vec<f32> = (0..(m * k)).map(|i| ((i as f32) * 0.07).sin()).collect();
    let w_data: Vec<f32> = (0..(k * n)).map(|i| ((i as f32) * 0.03).cos()).collect();

    let x_gpu = GpuTensor::from_f32_slice(&ctx, &x_batch, vec![m, k]).unwrap();
    let w_q4 = GpuQ4Tensor::from_f32_matrix(&ctx, &w_data, k, n).unwrap();

    // Execute single-pass M=5 batched Q4 micro-GEMM (reads W_q4 once for all 5 tokens)
    let y_batch_gpu = x_gpu.gemm_q4_batched(&ctx, &w_q4).unwrap();
    let y_batch_res = y_batch_gpu.to_vec(&ctx).unwrap();
    assert_eq!(y_batch_res.len(), m * n);

    // Verify each row of batched M=5 micro-GEMM matches individual M=1 gemv_q4 exactly
    for row in 0..m {
        let row_slice = &x_batch[(row * k)..((row + 1) * k)];
        let row_gpu = GpuTensor::from_f32_slice(&ctx, row_slice, vec![1, k]).unwrap();
        let row_out = row_gpu.gemv_q4(&ctx, &w_q4).unwrap().to_vec(&ctx).unwrap();

        for col in 0..n {
            let batched_val = y_batch_res[row * n + col];
            let single_val = row_out[col];
            assert!(
                (batched_val - single_val).abs() < 1e-4,
                "Row {}, Col {}: batched M=5 {} vs single M=1 {}",
                row,
                col,
                batched_val,
                single_val
            );
        }
    }
}

#[test]
fn test_native_gpu_fused_add_rmsnorm_accuracy() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test: {}", e);
            return;
        }
    };

    let num_tokens = 3;
    let hidden_dim = 64;
    let eps = 1e-6f32;

    let x_data: Vec<f32> = (0..(num_tokens * hidden_dim)).map(|i| (i as f32) * 0.05 - 1.0).collect();
    let res_data: Vec<f32> = (0..(num_tokens * hidden_dim)).map(|i| ((i as f32) * 0.1).cos()).collect();
    let w_data: Vec<f32> = (0..hidden_dim).map(|i| 0.5 + (i as f32) * 0.01).collect();

    let x_gpu = GpuTensor::from_f32_slice(&ctx, &x_data, vec![num_tokens, hidden_dim]).unwrap();
    let res_gpu = GpuTensor::from_f32_slice(&ctx, &res_data, vec![num_tokens, hidden_dim]).unwrap();
    let w_gpu = GpuTensor::from_f32_slice(&ctx, &w_data, vec![hidden_dim]).unwrap();

    let y_gpu = x_gpu.fused_add_rmsnorm(&ctx, &res_gpu, &w_gpu, eps).unwrap();
    let y_res = y_gpu.to_vec(&ctx).unwrap();

    for t in 0..num_tokens {
        let offset = t * hidden_dim;
        let mut sq_sum = 0.0f32;
        for i in 0..hidden_dim {
            let h = x_data[offset + i] + res_data[offset + i];
            sq_sum += h * h;
        }
        let scale = 1.0 / (sq_sum / (hidden_dim as f32) + eps).sqrt();
        for i in 0..hidden_dim {
            let h = x_data[offset + i] + res_data[offset + i];
            let expected = h * scale * w_data[i];
            let actual = y_res[offset + i];
            assert!(
                (actual - expected).abs() < 1e-4,
                "Token {}, Dim {}: expected {}, got {}",
                t,
                i,
                expected,
                actual
            );
        }
    }
}

#[test]
fn test_native_gpu_flash_attention_gqa_40_8() {
    let ctx = match GpuContext::new() {
        Ok(c) => c,
        Err(e) => {
            println!("Skipping GPU test: {}", e);
            return;
        }
    };

    // Exact Qwen2.5-Coder-32B head topology: 40 Q heads, 8 KV heads (5:1 GQA), head_dim = 128
    // Speculative verification batch: seq_len_q = 5 tokens attending over seq_len_kv = 9 cached KV tokens
    let seq_len_q = 5;
    let seq_len_kv = 9;
    let num_q_heads = 40;
    let num_kv_heads = 8;
    let head_dim = 128;
    let gqa_ratio = num_q_heads / num_kv_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();

    let q_len = seq_len_q * num_q_heads * head_dim;
    let kv_len = seq_len_kv * num_kv_heads * head_dim;

    let q_data: Vec<f32> = (0..q_len).map(|i| ((i as f32) * 0.013).sin() * 0.5).collect();
    let k_data: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.017).cos() * 0.5).collect();
    let v_data: Vec<f32> = (0..kv_len).map(|i| ((i as f32) * 0.019).sin() * 0.5).collect();

    let q_gpu = GpuTensor::from_f32_slice(&ctx, &q_data, vec![seq_len_q, num_q_heads, head_dim]).unwrap();
    let k_gpu = GpuTensor::from_f32_slice(&ctx, &k_data, vec![seq_len_kv, num_kv_heads, head_dim]).unwrap();
    let v_gpu = GpuTensor::from_f32_slice(&ctx, &v_data, vec![seq_len_kv, num_kv_heads, head_dim]).unwrap();

    let out_gpu = q_gpu
        .flash_attention_gqa(&ctx, &k_gpu, &v_gpu, num_q_heads, num_kv_heads, head_dim, true)
        .unwrap();
    let out_res = out_gpu.to_vec(&ctx).unwrap();
    assert_eq!(out_res.len(), q_len);

    // CPU reference with causal speculative mask: past_len = seq_len_kv - seq_len_q = 4
    let past_len = seq_len_kv - seq_len_q;
    for qi in 0..seq_len_q {
        let max_k_pos = past_len + qi + 1;
        for qh in 0..num_q_heads {
            let kv_h = qh / gqa_ratio;
            let q_base = (qi * num_q_heads + qh) * head_dim;

            let mut m_prev = -1e30f32;
            let mut l_prev = 0.0f32;
            let mut acc = vec![0.0f32; head_dim];

            for ki in 0..max_k_pos {
                let kv_base = (ki * num_kv_heads + kv_h) * head_dim;
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_data[q_base + d] * k_data[kv_base + d];
                }
                let s = dot * scale;
                let m_curr = m_prev.max(s);
                let exp_prev = (m_prev - m_curr).exp();
                let exp_curr = (s - m_curr).exp();
                l_prev = l_prev * exp_prev + exp_curr;
                for d in 0..head_dim {
                    acc[d] = acc[d] * exp_prev + exp_curr * v_data[kv_base + d];
                }
                m_prev = m_curr;
            }

            for d in 0..head_dim {
                let expected = acc[d] / l_prev;
                let actual = out_res[q_base + d];
                assert!(
                    (actual - expected).abs() < 1e-4,
                    "GQA FlashAttn mismatch at qi={}, qh={}, d={}: expected {}, got {}",
                    qi,
                    qh,
                    d,
                    expected,
                    actual
                );
            }
        }
    }
}



