use harness_core::{DType, Device, Result, Shape, Tensor};
use rayon::prelude::*;

// Standard 16 NF4 quantization bin centers
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

pub fn quantize_nf4(tensor: &Tensor) -> Result<(Vec<u8>, f32)> {
    let data = tensor.as_f32_slice()?;
    let max_abs = data.par_iter().map(|v| v.abs()).reduce(|| 0.0f32, |a, b| a.max(b));
    let scale = if max_abs > 1e-8 { max_abs } else { 1.0 };
    let inv_scale = 1.0 / scale;

    let num_bytes = (data.len() + 1) / 2;
    let mut packed = vec![0u8; num_bytes];

    packed.par_iter_mut().enumerate().for_each(|(byte_idx, byte)| {
        let i0 = byte_idx * 2;
        let i1 = i0 + 1;

        let find_closest_bin = |val: f32| -> u8 {
            let normalized = val * inv_scale;
            let mut best_bin = 0;
            let mut best_diff = f32::MAX;
            for (idx, &bin) in NF4_BINS.iter().enumerate() {
                let diff = (normalized - bin).abs();
                if diff < best_diff {
                    best_diff = diff;
                    best_bin = idx as u8;
                }
            }
            best_bin
        };

        let q0 = if i0 < data.len() { find_closest_bin(data[i0]) } else { 7 };
        let q1 = if i1 < data.len() { find_closest_bin(data[i1]) } else { 7 };

        *byte = q0 | (q1 << 4);
    });

    Ok((packed, scale))
}

pub fn dequantize_nf4(packed: &[u8], scale: f32, shape: Shape, device: Device) -> Result<Tensor> {
    let numel: usize = shape.iter().product();
    let mut out = vec![0.0f32; numel];

    out.par_chunks_mut(2).enumerate().for_each(|(chunk_idx, chunk)| {
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
