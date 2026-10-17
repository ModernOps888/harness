use harness_core::{Device, Result, Shape, Tensor};
use rayon::prelude::*;

pub const FP8_BLOCK_SIZE: usize = 32;
pub const FP8_E4M3_MAX: f32 = 448.0;

/// Encode an IEEE-754 f32 scalar into an OCP FP8 E4M3 byte:
/// [1 sign bit | 4 exponent bits (bias = 7) | 3 mantissa bits]
/// Normal range: +/- 2^(e-7) * (1 + m/8) for e in 1..=15 (max normal: e=15, m=6 -> 448.0)
/// Subnormal range: +/- 2^(-6) * (m/8) = +/- m * 2^(-9) for e=0, m in 1..=7
#[inline]
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    if !x.is_finite() || x == 0.0 {
        return 0;
    }
    let sign: u8 = if x < 0.0 { 0x80 } else { 0x00 };
    let abs_x = x.abs().min(FP8_E4M3_MAX);

    // Smallest positive normal is 2^-6 = 0.015625
    if abs_x < 0.015625 {
        // Subnormal representation: value = m * 2^-9 (step = 1/512)
        let m = (abs_x * 512.0).round() as u8;
        if m >= 8 {
            // Rounds up into smallest normal (e = 1, m = 0)
            return sign | (1 << 3);
        }
        return sign | (m & 0x07);
    }

    let e_unbiased = abs_x.log2().floor() as i32;
    let mut e = (e_unbiased + 7).clamp(1, 15) as u8;
    let base = (2.0f32).powi((e as i32) - 7);
    let mut m = (((abs_x / base) - 1.0) * 8.0).round() as u8;

    if m >= 8 {
        m = 0;
        e += 1;
    }
    // In OCP E4M3FN, e=15, m=7 (0x7F/0xFF) is reserved for NaN; clamp to max normal (e=15, m=6 = 448.0)
    if e > 15 || (e == 15 && m > 6) {
        e = 15;
        m = 6;
    }

    sign | (e << 3) | (m & 0x07)
}

/// Decode an OCP FP8 E4M3 byte back into an IEEE-754 f32 scalar
#[inline]
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((byte >> 3) & 0x0F) as i32;
    let mant = (byte & 0x07) as f32;

    if exp == 0 {
        if mant == 0.0 {
            0.0
        } else {
            // Subnormal: (-1)^s * 2^-6 * (m / 8)
            sign * mant * (1.0 / 512.0)
        }
    } else if exp == 15 && (byte & 0x07) == 7 {
        // E4M3FN NaN saturate to max normal for numerical stability in GEMM
        sign * FP8_E4M3_MAX
    } else {
        sign * (1.0 + mant * 0.125) * (2.0f32).powi(exp - 7)
    }
}

/// A 32-element group-scaled OCP FP8 E4M3 tensor preserving heavy-tailed outlier dynamic range
#[derive(Clone, Debug)]
pub struct Fp8Tensor {
    pub shape: Shape,
    pub data: Vec<u8>,
    pub group_scales: Vec<f32>,
    pub scale: f32,
    pub device: Device,
}

impl Fp8Tensor {
    /// Quantize an F32 tensor into 32-element group-scaled OCP FP8 E4M3 format
    pub fn from_f32_tensor(tensor: &Tensor) -> Result<Self> {
        let f32_data = tensor.as_f32_slice()?;
        let numel = f32_data.len();
        let num_groups = numel.div_ceil(FP8_BLOCK_SIZE);

        let mut data = vec![0u8; numel];
        let mut group_scales = vec![1.0f32; num_groups];

        data.par_chunks_mut(FP8_BLOCK_SIZE)
            .zip(group_scales.par_iter_mut())
            .enumerate()
            .for_each(|(g_idx, (out_chunk, g_scale))| {
                let start = g_idx * FP8_BLOCK_SIZE;
                let end = (start + FP8_BLOCK_SIZE).min(numel);
                let in_chunk = &f32_data[start..end];

                let max_abs = in_chunk.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
                let inv_scale = if max_abs > 1e-8 {
                    max_abs / FP8_E4M3_MAX
                } else {
                    1.0 / FP8_E4M3_MAX
                };
                let fwd_scale = 1.0 / inv_scale;
                *g_scale = inv_scale;

                for (byte, &val) in out_chunk.iter_mut().zip(in_chunk.iter()) {
                    *byte = f32_to_fp8_e4m3(val * fwd_scale);
                }
            });

        let mean_scale = if group_scales.is_empty() {
            1.0
        } else {
            group_scales.iter().sum::<f32>() / group_scales.len() as f32
        };

        Ok(Self {
            shape: tensor.shape().to_vec(),
            data,
            group_scales,
            scale: mean_scale,
            device: tensor.device().clone(),
        })
    }

    /// Dequantize 32-element group-scaled FP8 E4M3 tensor back to F32
    pub fn dequantize(&self) -> Result<Tensor> {
        let numel: usize = self.shape.iter().product();
        let mut out = vec![0.0f32; numel];

        out.par_chunks_mut(FP8_BLOCK_SIZE)
            .zip(self.data.par_chunks(FP8_BLOCK_SIZE))
            .zip(self.group_scales.par_iter())
            .for_each(|((out_chunk, in_chunk), &inv_scale)| {
                for (dst, &byte) in out_chunk.iter_mut().zip(in_chunk.iter()) {
                    *dst = fp8_e4m3_to_f32(byte) * inv_scale;
                }
            });

        Tensor::from_f32_slice(&out, self.shape.clone(), self.device.clone())
    }

    /// Memory footprint in bytes (1B per element + 4B per 32-element group scale)
    pub fn memory_size_bytes(&self) -> usize {
        self.data.len() + self.group_scales.len() * std::mem::size_of::<f32>()
    }
}

/// Quantize F32 tensor to real OCP FP8 (E4M3 representation with scale factor)
pub fn quantize_fp8(tensor: &Tensor) -> Result<(Vec<u8>, f32)> {
    let f32_data = tensor.as_f32_slice()?;
    let max_abs = f32_data
        .par_iter()
        .map(|v| v.abs())
        .reduce(|| 0.0f32, |a, b| a.max(b));

    // Map max_abs into E4M3 normal dynamic range (max normal = 448.0)
    let scale = if max_abs > 1e-8 {
        FP8_E4M3_MAX / max_abs
    } else {
        1.0
    };
    let mut out = vec![0u8; f32_data.len()];

    out.par_iter_mut()
        .zip(f32_data.par_iter())
        .for_each(|(byte, &val)| {
            *byte = f32_to_fp8_e4m3(val * scale);
        });

    Ok((out, 1.0 / scale))
}

/// Dequantize OCP FP8 E4M3 raw bytes back to F32
pub fn dequantize_fp8(
    bytes: &[u8],
    inv_scale: f32,
    shape: Shape,
    device: Device,
) -> Result<Tensor> {
    let mut out = vec![0.0f32; bytes.len()];
    out.par_iter_mut()
        .zip(bytes.par_iter())
        .for_each(|(val, &b)| {
            *val = fp8_e4m3_to_f32(b) * inv_scale;
        });

    Tensor::from_f32_slice(&out, shape, device)
}

