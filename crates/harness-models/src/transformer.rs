use harness_attention::{flash_attention_v3, FlashAttentionConfig, KVCache, RotaryEmbedding};
use harness_core::{Result, Tensor};

pub struct TransformerBlock {
    pub layer_idx: usize,
    pub input_norm_weight: Tensor,
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    pub post_norm_weight: Tensor,
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
    pub attn_config: FlashAttentionConfig,
    pub rope: RotaryEmbedding,
    pub eps: f32,
}

impl TransformerBlock {
    pub fn forward(
        &self,
        x: &Tensor,
        _start_pos: usize,
        _kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        // 1. Input RMSNorm
        let normed = x.rms_norm(&self.input_norm_weight, self.eps)?;

        // 2. QKV Linear Projections
        let q = normed.matmul(&self.q_proj)?;
        let k = normed.matmul(&self.k_proj)?;
        let v = normed.matmul(&self.v_proj)?;

        // 3. Flash Attention v3 (computes tiled attention)
        let attn_out = flash_attention_v3(&q, &k, &v, &self.attn_config)?;

        // 4. Output projection and residual connection
        let proj_out = attn_out.matmul(&self.o_proj)?;
        let h1 = self.add_residual(x, &proj_out)?;

        // 5. Post-attention RMSNorm
        let post_normed = h1.rms_norm(&self.post_norm_weight, self.eps)?;

        // 6. MLP with SwiGLU: (silu(x @ gate) * (x @ up)) @ down
        let gate = post_normed.matmul(&self.gate_proj)?;
        let up = post_normed.matmul(&self.up_proj)?;
        let swiglu = gate.silu_glu(&up)?;
        let mlp_out = swiglu.matmul(&self.down_proj)?;

        // 7. MLP residual connection
        self.add_residual(&h1, &mlp_out)
    }

    fn add_residual(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_slice = a.as_f32_slice()?;
        let b_slice = b.as_f32_slice()?;
        let mut out = vec![0.0f32; a_slice.len()];
        for i in 0..a_slice.len() {
            out[i] = a_slice[i] + b_slice[i];
        }
        Tensor::from_f32_slice(&out, a.shape().to_vec(), a.device().clone())
    }
}
