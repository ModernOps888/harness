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

    /// Activation-aware (`imatrix`) weighted Q4 quantization:
    /// Minimizes importance-weighted reconstruction error \sum_i I_i * (w_i - \hat{w}_i)^2
    /// across each 32-element block using coordinate search + weighted affine refinement.
    pub fn from_f32_tensor_imatrix(tensor: &Tensor, importance_weights: &[f32]) -> Result<Self> {
        let f32_data = tensor.as_f32_slice()?;
        let numel = f32_data.len();
        let num_blocks = numel.div_ceil(BLOCK_SIZE);
        let cols = *tensor.shape().last().unwrap_or(&numel.max(1));

        let mut blocks = vec![
            Q4Block {
                scale: 0.0,
                min_val: 0.0,
                qs: [0; 16],
            };
            num_blocks
        ];

        blocks
            .par_iter_mut()
            .enumerate()
            .for_each(|(b_idx, block)| {
                let start = b_idx * BLOCK_SIZE;
                let end = (start + BLOCK_SIZE).min(numel);
                let chunk = &f32_data[start..end];

                let mut imp = [1.0f32; BLOCK_SIZE];
                for (i, slot) in imp.iter_mut().enumerate().take(chunk.len()) {
                    let global_idx = start + i;
                    let w_imp = if importance_weights.len() == numel {
                        importance_weights[global_idx]
                    } else if !importance_weights.is_empty() {
                        importance_weights[(global_idx % cols) % importance_weights.len()]
                    } else {
                        1.0
                    };
                    *slot = w_imp.max(1e-6);
                }

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

                let range = max_val - min_val;
                if range <= 1e-8 {
                    block.scale = 1.0;
                    block.min_val = min_val;
                    block.qs = [0; 16];
                    return;
                }

                // Grid search over range shrinkage factors to protect high-importance outlier weights
                let mut best_loss = f32::MAX;
                let mut best_scale = range / 15.0;
                let mut best_min = min_val;

                for step in 0..=8 {
                    let shrink = 1.0 - (step as f32) * 0.02; // [1.00 .. 0.84]
                    // Compute importance-weighted center to anchor the shrunk interval
                    let mut w_sum = 0.0f32;
                    let mut iw_sum = 0.0f32;
                    for (idx, &v) in chunk.iter().enumerate() {
                        w_sum += imp[idx];
                        iw_sum += imp[idx] * v;
                    }
                    let center = if w_sum > 0.0 {
                        iw_sum / w_sum
                    } else {
                        0.5 * (min_val + max_val)
                    };

                    let cand_min = if step == 0 {
                        min_val
                    } else {
                        (center - (center - min_val) * shrink).max(min_val)
                    };
                    let cand_max = if step == 0 {
                        max_val
                    } else {
                        (center + (max_val - center) * shrink).min(max_val)
                    };
                    let cand_diff = (cand_max - cand_min).max(1e-8);
                    let mut cand_scale = cand_diff / 15.0;
                    let cand_min_refined = cand_min;

                    // 1 step of weighted least-squares scale refinement for fixed integer bins
                    let inv_s = 1.0 / cand_scale;
                    let mut sum_wq = 0.0f32;
                    let mut sum_qq = 0.0f32;
                    for (idx, &v) in chunk.iter().enumerate() {
                        let q = ((v - cand_min_refined) * inv_s).round().clamp(0.0, 15.0);
                        sum_wq += imp[idx] * (v - cand_min_refined) * q;
                        sum_qq += imp[idx] * q * q;
                    }
                    if sum_qq > 1e-8 {
                        cand_scale = (sum_wq / sum_qq).max(1e-8);
                    }

                    let inv_s2 = 1.0 / cand_scale;
                    let mut loss = 0.0f32;
                    for (idx, &v) in chunk.iter().enumerate() {
                        let q = ((v - cand_min_refined) * inv_s2).round().clamp(0.0, 15.0);
                        let rec = q * cand_scale + cand_min_refined;
                        let err = v - rec;
                        loss += imp[idx] * err * err;
                    }

                    if loss < best_loss {
                        best_loss = loss;
                        best_scale = cand_scale;
                        best_min = cand_min_refined;
                    }
                }

                let inv_scale = 1.0 / best_scale;
                block.scale = best_scale;
                block.min_val = best_min;

                for i in 0..16 {
                    let idx0 = i * 2;
                    let idx1 = idx0 + 1;

                    let q0 = if idx0 < chunk.len() {
                        (((chunk[idx0] - best_min) * inv_scale)
                            .round()
                            .clamp(0.0, 15.0)) as u8
                    } else {
                        0
                    };

                    let q1 = if idx1 < chunk.len() {
                        (((chunk[idx1] - best_min) * inv_scale)
                            .round()
                            .clamp(0.0, 15.0)) as u8
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

    /// Fused in-register matrix-vector multiplication (`y = W_q4 * x` or `y = x * W_q4`)
    /// directly from packed 4-bit nibbles without materializing a full `Vec<f32>` weight matrix in RAM.
    /// Uses algebraic block factoring: \sum_{j=0}^{31} (s * q_j + m) * x_j = s * \sum (q_j * x_j) + m * \sum x_j
    pub fn matvec_fused(&self, x: &[f32]) -> Result<Vec<f32>> {
        let (rows, cols) = match self.shape.as_slice() {
            [r, c] => (*r, *c),
            [c] => (1, *c),
            other => {
                return Err(harness_core::HarnessError::InvalidShape(format!(
                    "matvec_fused requires 1D or 2D Q4 tensor, found shape {:?}",
                    other
                )))
            }
        };

        if x.len() == cols {
            let mut y = vec![0.0f32; rows];
            if cols.is_multiple_of(BLOCK_SIZE) {
                let blocks_per_row = cols / BLOCK_SIZE;
                y.par_iter_mut().enumerate().for_each(|(r, y_out)| {
                    let row_blocks = &self.blocks[r * blocks_per_row..(r + 1) * blocks_per_row];
                    let mut row_acc = 0.0f32;
                    for (b_idx, block) in row_blocks.iter().enumerate() {
                        let x_chunk = &x[b_idx * BLOCK_SIZE..(b_idx + 1) * BLOCK_SIZE];
                        let mut dot_qx = 0.0f32;
                        let mut sum_x = 0.0f32;
                        for i in 0..16 {
                            let byte = block.qs[i];
                            let q0 = (byte & 0x0F) as f32;
                            let q1 = ((byte >> 4) & 0x0F) as f32;
                            let x0 = x_chunk[i * 2];
                            let x1 = x_chunk[i * 2 + 1];
                            dot_qx += q0 * x0 + q1 * x1;
                            sum_x += x0 + x1;
                        }
                        row_acc += block.scale * dot_qx + block.min_val * sum_x;
                    }
                    *y_out = row_acc;
                });
            } else {
                y.par_iter_mut().enumerate().for_each(|(r, y_out)| {
                    let mut row_acc = 0.0f32;
                    for (c, &xv) in x.iter().enumerate().take(cols) {
                        let flat_idx = r * cols + c;
                        let block = &self.blocks[flat_idx / BLOCK_SIZE];
                        let lane = flat_idx % BLOCK_SIZE;
                        let byte = block.qs[lane / 2];
                        let q = if lane.is_multiple_of(2) {
                            byte & 0x0F
                        } else {
                            (byte >> 4) & 0x0F
                        };
                        let w = (q as f32) * block.scale + block.min_val;
                        row_acc += w * xv;
                    }
                    *y_out = row_acc;
                });
            }
            Ok(y)
        } else if x.len() == rows {
            // Left multiplication: y[1, cols] = x[1, rows] * W[rows, cols]
            let mut y = vec![0.0f32; cols];
            for (r, &xv) in x.iter().enumerate().take(rows) {
                for (c, y_out) in y.iter_mut().enumerate().take(cols) {
                    let flat_idx = r * cols + c;
                    let block = &self.blocks[flat_idx / BLOCK_SIZE];
                    let lane = flat_idx % BLOCK_SIZE;
                    let byte = block.qs[lane / 2];
                    let q = if lane.is_multiple_of(2) {
                        byte & 0x0F
                    } else {
                        (byte >> 4) & 0x0F
                    };
                    let w = (q as f32) * block.scale + block.min_val;
                    *y_out += xv * w;
                }
            }
            Ok(y)
        } else {
            Err(harness_core::HarnessError::ShapeMismatch {
                expected: vec![cols],
                found: vec![x.len()],
            })
        }
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

