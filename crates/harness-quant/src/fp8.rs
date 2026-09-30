use harness_core::{DType, Device, Result, Shape, Tensor};
use rayon::prelude::*;

/// Quantize F32 tensor to FP8 (E4M3 representation with per-tensor scale)
pub fn quantize_fp8(tensor: &Tensor) -> Result<(Vec<u8>, f32)> {
    let f32_data = tensor.as_f32_slice()?;
    let max_abs = f32_data
        .par_iter()
        .map(|v| v.abs())
        .reduce(|| 0.0f32, |a, b| a.max(b));

    // E4M3 max representable value without NaN is ~448.0
    let scale = if max_abs > 1e-8 { 448.0 / max_abs } else { 1.0 };
    let mut out = vec![0u8; f32_data.len()];

    out.par_iter_mut()
        .zip(f32_data.par_iter())
        .for_each(|(byte, &val)| {
            let scaled = val * scale;
            let clamped = scaled.clamp(-448.0, 448.0);
            // Linear approximation quantizer
            let quantized = ((clamped / 448.0) * 127.0).round() as i8;
            *byte = quantized as u8;
        });

    Ok((out, 1.0 / scale))
}

/// Dequantize FP8 raw bytes back to F32
pub fn dequantize_fp8(bytes: &[u8], inv_scale: f32, shape: Shape, device: Device) -> Result<Tensor> {
    let mut out = vec![0.0f32; bytes.len()];
    out.par_iter_mut()
        .zip(bytes.par_iter())
        .for_each(|(val, &b)| {
            let signed = b as i8;
            *val = (signed as f32 / 127.0) * 448.0 * inv_scale;
        });

    Tensor::from_f32_slice(&out, shape, device)
}
