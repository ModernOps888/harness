use harness_core::{DType, Device, Result, Tensor};
use harness_quant::{f32_to_fp8_e4m3, fp8_e4m3_to_f32, FP8_E4M3_MAX};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheCompression {
    None,       // FP16/F32
    FP8E4M3,    // 50% memory reduction vs FP16 (75% vs F32)
    FP4,        // 75% memory reduction vs FP16 (87.5% vs F32)
}

pub struct KVCache {
    pub k: Tensor,
    pub v: Tensor,
    pub max_seq_len: usize,
    pub current_len: usize,
    pub compression: KVCacheCompression,
    group_size: usize,
    k_quant: Vec<u8>,
    v_quant: Vec<u8>,
    k_scales: Vec<f32>,
    v_scales: Vec<f32>,
    k_mins: Vec<f32>,
    v_mins: Vec<f32>,
}

impl KVCache {
    pub fn new(
        _num_layers: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
        device: Device,
        compression: KVCacheCompression,
    ) -> Result<Self> {
        let dtype = match compression {
            KVCacheCompression::None => DType::F32,
            KVCacheCompression::FP8E4M3 => DType::FP8E4M3,
            KVCacheCompression::FP4 => DType::FP4,
        };

        let shape = vec![max_seq_len, num_kv_heads, head_dim];
        let k = Tensor::zeros(shape.clone(), dtype, device.clone())?;
        let v = Tensor::zeros(shape, dtype, device)?;

        let total_elems = max_seq_len * num_kv_heads * head_dim;
        let token_stride = (num_kv_heads * head_dim).max(1);
        let group_size = 32usize.min(token_stride).max(1);
        let num_groups = total_elems.div_ceil(group_size);

        let (k_quant, v_quant) = match compression {
            KVCacheCompression::None => (Vec::new(), Vec::new()),
            KVCacheCompression::FP8E4M3 => (vec![0u8; total_elems], vec![0u8; total_elems]),
            KVCacheCompression::FP4 => {
                let bytes = total_elems.div_ceil(2);
                (vec![0u8; bytes], vec![0u8; bytes])
            }
        };

        let (k_scales, v_scales) = match compression {
            KVCacheCompression::None => (Vec::new(), Vec::new()),
            _ => (vec![1.0f32; num_groups], vec![1.0f32; num_groups]),
        };

        let (k_mins, v_mins) = match compression {
            KVCacheCompression::FP4 => (vec![0.0f32; num_groups], vec![0.0f32; num_groups]),
            _ => (Vec::new(), Vec::new()),
        };

        Ok(Self {
            k,
            v,
            max_seq_len,
            current_len: 0,
            compression,
            group_size,
            k_quant,
            v_quant,
            k_scales,
            v_scales,
            k_mins,
            v_mins,
        })
    }

    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<()> {
        let new_tokens = k_new.shape().first().copied().unwrap_or(0);
        if new_tokens == 0 {
            return Ok(());
        }
        if self.current_len + new_tokens > self.max_seq_len {
            return Err(harness_core::HarnessError::Attention(format!(
                "KV cache sequence limit exceeded: current {} + new {} > max {}",
                self.current_len, new_tokens, self.max_seq_len
            )));
        }

        let num_kv_heads = self.k.shape()[1];
        let head_dim = self.k.shape()[2];
        let token_stride = num_kv_heads * head_dim;

        let start_offset = self.current_len * token_stride;
        let copy_len = new_tokens * token_stride;

        let k_new_slice = k_new.as_f32_slice()?;
        let v_new_slice = v_new.as_f32_slice()?;

        match self.compression {
            KVCacheCompression::None => {
                let k_mut = self.k.as_mut_f32_slice()?;
                let v_mut = self.v.as_mut_f32_slice()?;

                if start_offset + copy_len <= k_mut.len() && copy_len <= k_new_slice.len() {
                    k_mut[start_offset..start_offset + copy_len]
                        .copy_from_slice(&k_new_slice[..copy_len]);
                }
                if start_offset + copy_len <= v_mut.len() && copy_len <= v_new_slice.len() {
                    v_mut[start_offset..start_offset + copy_len]
                        .copy_from_slice(&v_new_slice[..copy_len]);
                }
            }
            KVCacheCompression::FP8E4M3 => {
                Self::quantize_fp8_range(
                    &k_new_slice[..copy_len],
                    start_offset,
                    self.group_size,
                    &mut self.k_quant,
                    &mut self.k_scales,
                );
                Self::quantize_fp8_range(
                    &v_new_slice[..copy_len],
                    start_offset,
                    self.group_size,
                    &mut self.v_quant,
                    &mut self.v_scales,
                );
                let shape = self.k.shape().to_vec();
                let device = self.k.device().clone();
                self.k = Tensor::from_raw_bytes(
                    self.k_quant.clone(),
                    shape.clone(),
                    DType::FP8E4M3,
                    device.clone(),
                )?;
                self.v = Tensor::from_raw_bytes(
                    self.v_quant.clone(),
                    shape,
                    DType::FP8E4M3,
                    device,
                )?;
            }
            KVCacheCompression::FP4 => {
                Self::quantize_q4_range(
                    &k_new_slice[..copy_len],
                    start_offset,
                    self.group_size,
                    &mut self.k_quant,
                    &mut self.k_scales,
                    &mut self.k_mins,
                );
                Self::quantize_q4_range(
                    &v_new_slice[..copy_len],
                    start_offset,
                    self.group_size,
                    &mut self.v_quant,
                    &mut self.v_scales,
                    &mut self.v_mins,
                );
                let shape = self.k.shape().to_vec();
                let device = self.k.device().clone();
                self.k = Tensor::from_raw_bytes(
                    self.k_quant.clone(),
                    shape.clone(),
                    DType::FP4,
                    device.clone(),
                )?;
                self.v = Tensor::from_raw_bytes(
                    self.v_quant.clone(),
                    shape,
                    DType::FP4,
                    device,
                )?;
            }
        }

        self.current_len += new_tokens;
        Ok(())
    }

    /// Reconstruct active `[current_len, num_kv_heads, head_dim]` K and V tensors in F32 for attention
    pub fn get_active_kv(&self) -> Result<(Tensor, Tensor)> {
        let num_kv_heads = self.k.shape()[1];
        let head_dim = self.k.shape()[2];
        let active_elems = self.current_len * num_kv_heads * head_dim;
        let active_shape = vec![self.current_len, num_kv_heads, head_dim];
        let device = self.k.device().clone();

        match self.compression {
            KVCacheCompression::None => {
                let k_slice = &self.k.as_f32_slice()?[..active_elems];
                let v_slice = &self.v.as_f32_slice()?[..active_elems];
                Ok((
                    Tensor::from_f32_slice(k_slice, active_shape.clone(), device.clone())?,
                    Tensor::from_f32_slice(v_slice, active_shape, device)?,
                ))
            }
            KVCacheCompression::FP8E4M3 => {
                let mut k_f32 = vec![0.0f32; active_elems];
                let mut v_f32 = vec![0.0f32; active_elems];
                for i in 0..active_elems {
                    let g = i / self.group_size;
                    k_f32[i] = fp8_e4m3_to_f32(self.k_quant[i]) * self.k_scales[g];
                    v_f32[i] = fp8_e4m3_to_f32(self.v_quant[i]) * self.v_scales[g];
                }
                Ok((
                    Tensor::from_f32_slice(&k_f32, active_shape.clone(), device.clone())?,
                    Tensor::from_f32_slice(&v_f32, active_shape, device)?,
                ))
            }
            KVCacheCompression::FP4 => {
                let mut k_f32 = vec![0.0f32; active_elems];
                let mut v_f32 = vec![0.0f32; active_elems];
                for i in 0..active_elems {
                    let g = i / self.group_size;
                    let byte_idx = i / 2;
                    let shift = (i % 2) * 4;
                    let qk = ((self.k_quant[byte_idx] >> shift) & 0x0F) as f32;
                    let qv = ((self.v_quant[byte_idx] >> shift) & 0x0F) as f32;
                    k_f32[i] = qk * self.k_scales[g] + self.k_mins[g];
                    v_f32[i] = qv * self.v_scales[g] + self.v_mins[g];
                }
                Ok((
                    Tensor::from_f32_slice(&k_f32, active_shape.clone(), device.clone())?,
                    Tensor::from_f32_slice(&v_f32, active_shape, device)?,
                ))
            }
        }
    }

    /// Actual compressed KV buffer footprint in bytes
    pub fn memory_bytes(&self) -> usize {
        let total_elems: usize = self.k.shape().iter().product();
        2 * self.k.dtype().byte_size_for_elements(total_elems)
    }

    fn quantize_fp8_range(
        src: &[f32],
        start_offset: usize,
        group_size: usize,
        dst_bytes: &mut [u8],
        dst_scales: &mut [f32],
    ) {
        let mut idx = 0;
        while idx < src.len() {
            let global_pos = start_offset + idx;
            let g_idx = global_pos / group_size;
            let rem_in_group = group_size - (global_pos % group_size);
            let chunk_len = rem_in_group.min(src.len() - idx);
            let chunk = &src[idx..idx + chunk_len];

            let max_abs = chunk.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let inv_scale = if max_abs > 1e-8 {
                max_abs / FP8_E4M3_MAX
            } else {
                1.0 / FP8_E4M3_MAX
            };
            let fwd_scale = 1.0 / inv_scale;
            if g_idx < dst_scales.len() {
                dst_scales[g_idx] = inv_scale;
            }

            for (j, &val) in chunk.iter().enumerate() {
                dst_bytes[global_pos + j] = f32_to_fp8_e4m3(val * fwd_scale);
            }
            idx += chunk_len;
        }
    }

    fn quantize_q4_range(
        src: &[f32],
        start_offset: usize,
        group_size: usize,
        dst_bytes: &mut [u8],
        dst_scales: &mut [f32],
        dst_mins: &mut [f32],
    ) {
        let mut idx = 0;
        while idx < src.len() {
            let global_pos = start_offset + idx;
            let g_idx = global_pos / group_size;
            let rem_in_group = group_size - (global_pos % group_size);
            let chunk_len = rem_in_group.min(src.len() - idx);
            let chunk = &src[idx..idx + chunk_len];

            let mut min_v = f32::MAX;
            let mut max_v = f32::MIN;
            for &v in chunk {
                if v < min_v {
                    min_v = v;
                }
                if v > max_v {
                    max_v = v;
                }
            }
            let diff = max_v - min_v;
            let scale = if diff > 1e-8 { diff / 15.0 } else { 1.0 };
            let inv_scale = 1.0 / scale;

            if g_idx < dst_scales.len() {
                dst_scales[g_idx] = scale;
                dst_mins[g_idx] = min_v;
            }

            for (j, &val) in chunk.iter().enumerate() {
                let pos = global_pos + j;
                let q = (((val - min_v) * inv_scale).round().clamp(0.0, 15.0)) as u8;
                let byte_idx = pos / 2;
                if pos.is_multiple_of(2) {
                    dst_bytes[byte_idx] = (dst_bytes[byte_idx] & 0xF0) | q;
                } else {
                    dst_bytes[byte_idx] = (dst_bytes[byte_idx] & 0x0F) | (q << 4);
                }
            }
            idx += chunk_len;
        }
    }
}

