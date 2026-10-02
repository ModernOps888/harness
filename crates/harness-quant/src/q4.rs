use harness_core::{Device, Result, Shape, Tensor};
use rayon::prelude::*;

pub const BLOCK_SIZE: usize = 32;

/// A single 32-element Q4 quantized block
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct Q4Block {
    pub scale: f32,
    pub min_val: f32,
    pub qs: [u8; BLOCK_SIZE / 2], // 16 bytes storing 32 x 4-bit nibbles
}

#[derive(Clone)]
pub struct QuantizedQ4Tensor {
    pub shape: Shape,
    pub blocks: Vec<Q4Block>,
    pub device: Device,
}

impl QuantizedQ4Tensor {
    /// Quantize an F32 tensor to Q4 blocks with per-block scale and min
    pub fn from_f32_tensor(tensor: &Tensor) -> Result<Self> {
        let f32_data = tensor.as_f32_slice()?;
        let numel = f32_data.len();
        let num_blocks = numel.div_ceil(BLOCK_SIZE);

        let mut blocks = vec![
            Q4Block {
                scale: 0.0,
                min_val: 0.0,
                qs: [0; 16],
            };
            num_blocks
        ];

        blocks.par_iter_mut().enumerate().for_each(|(b_idx, block)| {
            let start = b_idx * BLOCK_SIZE;
            let end = (start + BLOCK_SIZE).min(numel);
            let chunk = &f32_data[start..end];

            let mut min_val = f32::MAX;
            let mut max_val = f32::MIN;
            for &val in chunk {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }

            let diff = max_val - min_val;
            let scale = if diff > 1e-8 { diff / 15.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            block.scale = scale;
            block.min_val = min_val;

            for i in 0..16 {
                let idx0 = i * 2;
                let idx1 = idx0 + 1;

                let q0 = if idx0 < chunk.len() {
                    (((chunk[idx0] - min_val) * inv_scale).round() as u8).min(15)
                } else {
                    0
                };

                let q1 = if idx1 < chunk.len() {
                    (((chunk[idx1] - min_val) * inv_scale).round() as u8).min(15)
                } else {
                    0
                };

                block.qs[i] = q0 | (q1 << 4);
            }
        });

        Ok(Self {
            shape: tensor.shape().to_vec(),
            blocks,
            device: tensor.device().clone(),
        })
    }

    /// Fast dequantization to F32 for inference computation
    pub fn dequantize(&self) -> Result<Tensor> {
        let numel: usize = self.shape.iter().product();
        let mut out = vec![0.0f32; numel];

        out.par_chunks_mut(BLOCK_SIZE)
            .zip(self.blocks.par_iter())
            .for_each(|(out_chunk, block)| {
                let scale = block.scale;
                let min = block.min_val;
                for i in 0..16 {
                    let byte = block.qs[i];
                    let q0 = byte & 0x0F;
                    let q1 = (byte >> 4) & 0x0F;

                    let o0 = i * 2;
                    let o1 = o0 + 1;

                    if o0 < out_chunk.len() {
                        out_chunk[o0] = (q0 as f32) * scale + min;
                    }
                    if o1 < out_chunk.len() {
                        out_chunk[o1] = (q1 as f32) * scale + min;
                    }
                }
            });

        Tensor::from_f32_slice(&out, self.shape.clone(), self.device.clone())
    }

    /// Total memory size in bytes
    pub fn memory_size_bytes(&self) -> usize {
        self.blocks.len() * std::mem::size_of::<Q4Block>()
    }
}
