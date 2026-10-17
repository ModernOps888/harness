use harness_core::{Device, Result, Shape, Tensor};
use rayon::prelude::*;

pub const NF4_GROUP_SIZE: usize = 64;

// Standard 16 NF4 quantization bin centers (QLoRA information-theoretically optimal normal quantiles)
#[allow(clippy::excessive_precision)]
pub const NF4_BINS: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

// Exact 15 Voronoi decision boundaries: m[i] = 0.5 * (NF4_BINS[i] + NF4_BINS[i + 1])
#[allow(clippy::excessive_precision)]
pub const NF4_MIDPOINTS: [f32; 15] = [
    -0.8480964004993439,
    -0.6106329262256622,
    -0.4599952697753906,
    -0.33967943489551544,
    -0.23460740596055984,
    -0.13791173323988914,
    -0.045525018125772476,
    0.03979014977812767,
    0.1202552504837513,
    0.2035212516784668,
    0.2920137718319893,
    0.3893125355243683,
    0.5016634166240692,
    0.6427869200706482,
    0.8614784181118011,
];

/// Fast 4-step branchless binary search over the 15 Voronoi midpoints
#[inline(always)]
pub fn find_closest_nf4_bin(normalized: f32) -> u8 {
    let mut idx = 0usize;
    idx += usize::from(normalized > NF4_MIDPOINTS[idx + 7]) * 8;
    idx += usize::from(normalized > NF4_MIDPOINTS[idx + 3]) * 4;
    idx += usize::from(normalized > NF4_MIDPOINTS[idx + 1]) * 2;
    idx += usize::from(normalized > NF4_MIDPOINTS[idx]);
    idx as u8
}

/// Group-wise NF4 quantized tensor with 64-element absmax scales
#[derive(Clone, Debug)]
pub struct QuantizedNF4Tensor {
    pub shape: Shape,
    pub packed: Vec<u8>,
    pub scales: Vec<f32>,
    pub scale: f32,
    pub device: Device,
}

pub type Nf4Tensor = QuantizedNF4Tensor;

impl QuantizedNF4Tensor {
    /// Quantize an F32 tensor into 64-element group-scaled NF4 format
    pub fn from_f32_tensor(tensor: &Tensor) -> Result<Self> {
        let (packed, scales, scale) = quantize_nf4_grouped(tensor)?;
        Ok(Self {
            shape: tensor.shape().to_vec(),
            packed,
            scales,
            scale,
            device: tensor.device().clone(),
        })
    }

    /// Dequantize 64-element group-scaled NF4 tensor back to F32
    pub fn dequantize(&self) -> Result<Tensor> {
        dequantize_nf4_grouped(&self.packed, &self.scales, self.shape.clone(), self.device.clone())
    }

    /// Total memory footprint in bytes (0.5B per element + 4B per 64-element group scale)
    pub fn memory_size_bytes(&self) -> usize {
        self.packed.len() + self.scales.len() * std::mem::size_of::<f32>()
    }
}

/// Quantize an F32 tensor using 64-element group-wise absmax scales
pub fn quantize_nf4_grouped(tensor: &Tensor) -> Result<(Vec<u8>, Vec<f32>, f32)> {
    let data = tensor.as_f32_slice()?;
    let numel = data.len();
    let num_groups = numel.div_ceil(NF4_GROUP_SIZE);
    let mut scales = vec![1.0f32; num_groups];

    scales.par_iter_mut().enumerate().for_each(|(g_idx, s)| {
        let start = g_idx * NF4_GROUP_SIZE;
        let end = (start + NF4_GROUP_SIZE).min(numel);
        let max_abs = data[start..end]
            .iter()
            .fold(0.0f32, |acc, &v| acc.max(v.abs()));
        *s = if max_abs > 1e-8 { max_abs } else { 1.0 };
    });

    let global_scale = if scales.is_empty() {
        1.0
    } else {
        scales.iter().sum::<f32>() / scales.len() as f32
    };

    let num_bytes = numel.div_ceil(2);
    let mut packed = vec![0u8; num_bytes];

    packed
        .par_iter_mut()
        .enumerate()
        .for_each(|(byte_idx, byte)| {
            let i0 = byte_idx * 2;
            let i1 = i0 + 1;

            let q0 = if i0 < numel {
                let s0 = scales[i0 / NF4_GROUP_SIZE];
                find_closest_nf4_bin(data[i0] / s0)
            } else {
                7
            };
            let q1 = if i1 < numel {
                let s1 = scales[i1 / NF4_GROUP_SIZE];
                find_closest_nf4_bin(data[i1] / s1)
            } else {
                7
            };

            *byte = q0 | (q1 << 4);
        });

    Ok((packed, scales, global_scale))
}

/// Dequantize 64-element group-wise NF4 tensor back to F32
pub fn dequantize_nf4_grouped(
    packed: &[u8],
    scales: &[f32],
    shape: Shape,
    device: Device,
) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let mut out = vec![0.0f32; numel];

    out.par_chunks_mut(2)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let byte = packed[chunk_idx];
            let q0 = (byte & 0x0F) as usize;
            let q1 = ((byte >> 4) & 0x0F) as usize;

            let i0 = chunk_idx * 2;
            let i1 = i0 + 1;

            if !chunk.is_empty() {
                let s0 = scales.get(i0 / NF4_GROUP_SIZE).copied().unwrap_or(1.0);
                chunk[0] = NF4_BINS[q0] * s0;
            }
            if chunk.len() > 1 {
                let s1 = scales.get(i1 / NF4_GROUP_SIZE).copied().unwrap_or(1.0);
                chunk[1] = NF4_BINS[q1] * s1;
            }
        });

    Tensor::from_f32_slice(&out, shape, device)
}

pub fn quantize_nf4(tensor: &Tensor) -> Result<(Vec<u8>, f32)> {
    let data = tensor.as_f32_slice()?;
    let max_abs = data
        .par_iter()
        .map(|v| v.abs())
        .reduce(|| 0.0f32, |a, b| a.max(b));
    let scale = if max_abs > 1e-8 { max_abs } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let num_bytes = data.len().div_ceil(2);
    let mut packed = vec![0u8; num_bytes];

    packed
        .par_iter_mut()
        .enumerate()
        .for_each(|(byte_idx, byte)| {
            let i0 = byte_idx * 2;
            let i1 = i0 + 1;

            let q0 = if i0 < data.len() {
                find_closest_nf4_bin(data[i0] * inv_scale)
            } else {
                7
            };
            let q1 = if i1 < data.len() {
                find_closest_nf4_bin(data[i1] * inv_scale)
            } else {
                7
            };

            *byte = q0 | (q1 << 4);
        });

    Ok((packed, scale))
}

pub fn dequantize_nf4(packed: &[u8], scale: f32, shape: Shape, device: Device) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let mut out = vec![0.0f32; numel];

    out.par_chunks_mut(2)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let byte = packed[chunk_idx];
            let q0 = (byte & 0x0F) as usize;
            let q1 = ((byte >> 4) & 0x0F) as usize;

            if !chunk.is_empty() {
                chunk[0] = NF4_BINS[q0] * scale;
            }
            if chunk.len() > 1 {
                chunk[1] = NF4_BINS[q1] * scale;
            }
        });

    Tensor::from_f32_slice(&out, shape, device)
}

