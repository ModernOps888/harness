use harness_core::{Device, Result, Tensor};

/// Multi-Head Latent Attention (MLA) Engine:
/// Compresses Key-Value vectors into a shared low-rank latent vector (e.g. 512 dim),
/// reducing KV cache memory footprint by 90-95% compared to standard MHA.
pub struct MultiHeadLatentAttention {
    pub hidden_dim: usize,
    pub latent_dim_kv: usize, // e.g. 512 vs 4096 (8x compression)
    pub num_heads: usize,
    pub head_dim: usize,
    pub w_down_kv: Tensor,    // Down-projection: [hidden_dim, latent_dim_kv]
    pub w_up_k: Tensor,       // Up-projection: [latent_dim_kv, num_heads * head_dim]
    pub w_up_v: Tensor,       // Up-projection: [latent_dim_kv, num_heads * head_dim]
}

impl MultiHeadLatentAttention {
    pub fn new(
        hidden_dim: usize,
        latent_dim_kv: usize,
        num_heads: usize,
        head_dim: usize,
        device: &Device,
    ) -> Result<Self> {
        let w_down_kv = Tensor::zeros(vec![hidden_dim, latent_dim_kv], harness_core::DType::F32, device.clone())?;
        let w_up_k = Tensor::zeros(vec![latent_dim_kv, num_heads * head_dim], harness_core::DType::F32, device.clone())?;
        let w_up_v = Tensor::zeros(vec![latent_dim_kv, num_heads * head_dim], harness_core::DType::F32, device.clone())?;

        Ok(Self {
            hidden_dim,
            latent_dim_kv,
            num_heads,
            head_dim,
            w_down_kv,
            w_up_k,
            w_up_v,
        })
    }

    /// Compress input hidden state into compact low-rank KV latent representation to store in cache
    pub fn compress_kv(&self, x: &Tensor) -> Result<Tensor> {
        x.matmul(&self.w_down_kv)
    }

    /// Reconstruct full K and V vectors on-demand during decode attention
    pub fn decompress_kv(&self, latent_kv: &Tensor) -> Result<(Tensor, Tensor)> {
        let k = latent_kv.matmul(&self.w_up_k)?;
        let v = latent_kv.matmul(&self.w_up_v)?;
        Ok((k, v))
    }

    /// Memory compression ratio achieved by MLA (typically 0.88 - 0.95 = 90%+ VRAM saved)
    pub fn compression_ratio(&self) -> f32 {
        let standard_kv_size = 2 * self.num_heads * self.head_dim;
        let mla_size = self.latent_dim_kv;
        1.0 - (mla_size as f32 / standard_kv_size as f32)
    }
}
