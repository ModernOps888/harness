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
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        // 1. Input RMSNorm
        let normed = x.rms_norm(&self.input_norm_weight, self.eps)?;

        // 2. QKV Linear Projections
        let mut q = normed.matmul(&self.q_proj)?;
        let mut k = normed.matmul(&self.k_proj)?;
        let v = normed.matmul(&self.v_proj)?;

        let num_tokens = x.shape().first().copied().unwrap_or(1);
        let num_q_heads = self.attn_config.num_q_heads;
        let num_kv_heads = self.attn_config.num_kv_heads;
        let head_dim = self.attn_config.head_dim;

        // Apply Rotary Position Embeddings (RoPE)
        self.rope.apply(q.as_mut_f32_slice()?, start_pos, num_tokens, num_q_heads);
        self.rope.apply(k.as_mut_f32_slice()?, start_pos, num_tokens, num_kv_heads);

        // Reshape 2D [tokens, heads * dim] to 3D [tokens, heads, dim] for FlashAttention
        let q_3d = if q.shape().len() == 2 {
            q.reshape(vec![num_tokens, num_q_heads, head_dim])?
        } else {
            q
        };
        let k_3d = if k.shape().len() == 2 {
            k.reshape(vec![num_tokens, num_kv_heads, head_dim])?
        } else {
            k
        };
        let v_3d = if v.shape().len() == 2 {
            v.reshape(vec![num_tokens, num_kv_heads, head_dim])?
        } else {
            v
        };

        // Cache Key & Value representations
        kv_cache.append(&k_3d, &v_3d)?;

        // 3. Flash Attention v3 (computes tiled attention)
        let attn_out_3d = flash_attention_v3(&q_3d, &k_3d, &v_3d, &self.attn_config)?;
        let attn_out = attn_out_3d.reshape(vec![num_tokens, num_q_heads * head_dim])?;

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
