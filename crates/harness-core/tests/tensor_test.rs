use harness_core::{Device, Tensor};

#[test]
fn test_gemm_simd_unrolled_accuracy() {
    let device = Device::Cpu;

    // Test non-multiples of 4 to thoroughly exercise both the 4-wide SIMD loop and the tail loop
    let m = 7;
    let k = 5;
    let n = 9;

    let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.5) - 2.0).collect();
    let b_data: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.25) + 1.0).collect();

    let a = Tensor::from_f32_slice(&a_data, vec![m, k], device.clone()).unwrap();
    let b = Tensor::from_f32_slice(&b_data, vec![k, n], device.clone()).unwrap();

    let c = a.matmul(&b).unwrap();
    assert_eq!(c.shape(), &[m, n]);

    let c_slice = c.as_f32_slice().unwrap();

    // Verify against naive ground truth
    for i in 0..m {
        for j in 0..n {
            let mut expected = 0.0f32;
            for p in 0..k {
                expected += a_data[i * k + p] * b_data[p * n + j];
            }
            let actual = c_slice[i * n + j];
            assert!(
                (actual - expected).abs() < 1e-4,
                "GEMM mismatch at [{}, {}]: actual={}, expected={}",
                i, j, actual, expected
            );
        }
    }
}

#[test]
fn test_rms_norm_exactness() {
    let device = Device::Cpu;
    let hidden_dim = 8;
    let num_tokens = 2;

    let x_data = vec![
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // token 1
        -1.0, -2.0, -1.0, 2.0, 1.0, -2.0, 0.0, 1.0, // token 2
    ];
    let w_data = vec![1.0; hidden_dim]; // unit weights

    let x = Tensor::from_f32_slice(&x_data, vec![num_tokens, hidden_dim], device.clone()).unwrap();
    let w = Tensor::from_f32_slice(&w_data, vec![hidden_dim], device.clone()).unwrap();

    let eps = 1e-5;
    let norm = x.rms_norm(&w, eps).unwrap();
    let norm_slice = norm.as_f32_slice().unwrap();

    // Token 1 RMS check
    let sum_sq_1: f32 = x_data[0..8].iter().map(|v| v * v).sum();
    let rms_1 = 1.0 / (sum_sq_1 / 8.0 + eps).sqrt();
    for i in 0..8 {
        let expected = x_data[i] * rms_1;
        assert!((norm_slice[i] - expected).abs() < 1e-5);
    }
}

#[test]
fn test_silu_glu_accuracy() {
    let device = Device::Cpu;
    let x_data = vec![0.0, 1.0, -1.0, 2.0];
    let g_data = vec![2.0, 0.5, 3.0, -1.5];

    let x = Tensor::from_f32_slice(&x_data, vec![4], device.clone()).unwrap();
    let g = Tensor::from_f32_slice(&g_data, vec![4], device.clone()).unwrap();

    let out = x.silu_glu(&g).unwrap();
    let out_slice = out.as_f32_slice().unwrap();

    for i in 0..4 {
        let xi = x_data[i];
        let gi = g_data[i];
        let silu = xi / (1.0 + (-xi).exp());
        let expected = silu * gi;
        assert!((out_slice[i] - expected).abs() < 1e-5);
    }
}
