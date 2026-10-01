use harness_core::{DType, Device, Result, Tensor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KVCacheCompression {
    None,       // FP16/F32
    FP8E4M3,    // 50% memory reduction
    FP4,        // 75% memory reduction
}

pub struct KVCache {
    pub k: Tensor,
    pub v: Tensor,
    pub max_seq_len: usize,
    pub current_len: usize,
    pub compression: KVCacheCompression,
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

        Ok(Self {
            k,
            v,
            max_seq_len,
            current_len: 0,
            compression,
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

        let k_mut = self.k.as_mut_f32_slice()?;
        let v_mut = self.v.as_mut_f32_slice()?;

        if start_offset + copy_len <= k_mut.len() && copy_len <= k_new_slice.len() {
            k_mut[start_offset..start_offset + copy_len].copy_from_slice(&k_new_slice[..copy_len]);
        }
        if start_offset + copy_len <= v_mut.len() && copy_len <= v_new_slice.len() {
            v_mut[start_offset..start_offset + copy_len].copy_from_slice(&v_new_slice[..copy_len]);
        }

        self.current_len += new_tokens;
        Ok(())
    }
}
