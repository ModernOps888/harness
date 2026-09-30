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

    pub fn append(&mut self, k_new: &Tensor, _v_new: &Tensor) -> Result<()> {
        let new_tokens = k_new.shape()[0];
        if self.current_len + new_tokens > self.max_seq_len {
            return Err(harness_core::HarnessError::Attention(
                "KV cache sequence limit exceeded".into(),
            ));
        }
        self.current_len += new_tokens;
        Ok(())
    }
}
