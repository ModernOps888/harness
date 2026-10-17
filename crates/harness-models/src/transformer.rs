use harness_attention::{flash_attention_v3, FlashAttentionConfig, KVCache, RotaryEmbedding};
use harness_core::{Result, Tensor};

#[allow(dead_code)]
trait KVCacheActiveExt {
    fn get_active_kv(&self) -> Result<(Tensor, Tensor)>;
}

impl KVCacheActiveExt for KVCache {
    fn get_active_kv(&self) -> Result<(Tensor, Tensor)> {
        let active_len = self.current_len;
        let num_kv_heads = self.k.shape().get(1).copied().unwrap_or(1);
        let head_dim = self.k.shape().get(2).copied().unwrap_or(1);
        let total_elems = active_len * num_kv_heads * head_dim;

        let k_slice = self.k.as_f32_slice()?;
        let v_slice = self.v.as_f32_slice()?;

        let k_active = Tensor::from_f32_slice(
            &k_slice[..total_elems.min(k_slice.len())],
            vec![active_len, num_kv_heads, head_dim],
            self.k.device().clone(),
        )?;
        let v_active = Tensor::from_f32_slice(
            &v_slice[..total_elems.min(v_slice.len())],
            vec![active_len, num_kv_heads, head_dim],
            self.v.device().clone(),
        )?;
        Ok((k_active, v_active))
    }
}

pub struct TransformerBlock {
    pub layer_idx: usize,
    pub input_norm_weight: Tensor,
    pub q_proj: Tensor,
    pub k_proj: Tensor,
    pub v_proj: Tensor,
    pub o_proj: Tensor,
    /// Optional QKV attention biases (required by Qwen2.5-Coder-32B / 14B / 7B / 1.5B)
    pub q_bias: Option<Tensor>,
    pub k_bias: Option<Tensor>,
    pub v_bias: Option<Tensor>,
    /// Optional per-head QK-RMSNorm weights [head_dim] (required by Qwen3-32B / Qwen3-30B-A3B)
    pub q_norm: Option<Tensor>,
    pub k_norm: Option<Tensor>,
    pub post_norm_weight: Tensor,
    pub gate_proj: Tensor,
    pub up_proj: Tensor,
    pub down_proj: Tensor,
    pub attn_config: FlashAttentionConfig,
    pub rope: RotaryEmbedding,
    pub eps: f32,
}

impl TransformerBlock {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        layer_idx: usize,
        input_norm_weight: Tensor,
        q_proj: Tensor,
        k_proj: Tensor,
        v_proj: Tensor,
        o_proj: Tensor,
        post_norm_weight: Tensor,
        gate_proj: Tensor,
        up_proj: Tensor,
        down_proj: Tensor,
        attn_config: FlashAttentionConfig,
        rope: RotaryEmbedding,
        eps: f32,
    ) -> Self {
        Self {
            layer_idx,
            input_norm_weight,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_bias: None,
            k_bias: None,
            v_bias: None,
            q_norm: None,
            k_norm: None,
            post_norm_weight,
            gate_proj,
            up_proj,
            down_proj,
            attn_config,
            rope,
            eps,
        }
    }

    /// Attach QKV attention biases (used by Qwen2.5-Coder-32B architecture)
    pub fn with_qkv_bias(mut self, q_bias: Tensor, k_bias: Tensor, v_bias: Tensor) -> Self {
        self.q_bias = Some(q_bias);
        self.k_bias = Some(k_bias);
        self.v_bias = Some(v_bias);
        self
    }

    /// Attach per-head QK-RMSNorm weights (used by Qwen3-32B and Qwen3-30B-A3B MoE)
    pub fn with_qk_norm(mut self, q_norm: Tensor, k_norm: Tensor) -> Self {
        self.q_norm = Some(q_norm);
        self.k_norm = Some(k_norm);
        self
    }

    pub fn forward(
        &self,
        x: &Tensor,
        start_pos: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        // 1. Input RMSNorm
        let normed = x.rms_norm(&self.input_norm_weight, self.eps)?;

        // 2. QKV Linear Projections + Optional QKV Biases (Qwen2.5-Coder)
        let mut q = normed.matmul(&self.q_proj)?;
        let mut k = normed.matmul(&self.k_proj)?;
        let mut v = normed.matmul(&self.v_proj)?;

        if let Some(ref b) = self.q_bias {
            q = Self::add_bias_2d(&q, b)?;
        }
        if let Some(ref b) = self.k_bias {
            k = Self::add_bias_2d(&k, b)?;
        }
        if let Some(ref b) = self.v_bias {
            v = Self::add_bias_2d(&v, b)?;
        }

        let num_tokens = x.shape().first().copied().unwrap_or(1);
        let num_q_heads = self.attn_config.num_q_heads;
        let num_kv_heads = self.attn_config.num_kv_heads;
        let head_dim = self.attn_config.head_dim;

        // 3. Optional Per-Head QK-RMSNorm before RoPE (Qwen3-32B / Qwen3-30B-A3B)
        if let Some(ref qn) = self.q_norm {
            Self::apply_per_head_rmsnorm(
                q.as_mut_f32_slice()?,
                qn.as_f32_slice()?,
                num_tokens,
                num_q_heads,
                head_dim,
                self.eps,
            );
        }
        if let Some(ref kn) = self.k_norm {
            Self::apply_per_head_rmsnorm(
                k.as_mut_f32_slice()?,
                kn.as_f32_slice()?,
                num_tokens,
                num_kv_heads,
                head_dim,
                self.eps,
            );
        }

        // 4. Apply Rotary Position Embeddings (RoPE) with distinct GQA head counts (e.g. 40:8)
        self.rope
            .apply(q.as_mut_f32_slice()?, start_pos, num_tokens, num_q_heads);
        self.rope
            .apply(k.as_mut_f32_slice()?, start_pos, num_tokens, num_kv_heads);

        // Reshape 2D [tokens, heads * dim] to 3D [tokens, heads, dim] for KV cache & FlashAttention
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

        // 5. Append new K/V states and retrieve full accumulated KV history for attention
        kv_cache.append(&k_3d, &v_3d)?;
        let (k_full, v_full) = kv_cache.get_active_kv()?;

        // 6. Flash Attention v3 over full accumulated KV history
        let attn_out_3d = flash_attention_v3(&q_3d, &k_full, &v_full, &self.attn_config)?;
        let attn_out = attn_out_3d.reshape(vec![num_tokens, num_q_heads * head_dim])?;

        // 7. Output projection and residual connection
        let proj_out = attn_out.matmul(&self.o_proj)?;
        let h1 = self.add_residual(x, &proj_out)?;

        // 8. Post-attention RMSNorm
        let post_normed = h1.rms_norm(&self.post_norm_weight, self.eps)?;

        // 9. MLP with SwiGLU: (silu(x @ gate) * (x @ up)) @ down
        let gate = post_normed.matmul(&self.gate_proj)?;
        let up = post_normed.matmul(&self.up_proj)?;
        let swiglu = gate.silu_glu(&up)?;
        let mlp_out = swiglu.matmul(&self.down_proj)?;

        // 10. MLP residual connection
        self.add_residual(&h1, &mlp_out)
    }

    fn add_bias_2d(tensor: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let t_slice = tensor.as_f32_slice()?;
        let b_slice = bias.as_f32_slice()?;
        let cols = b_slice.len().max(1);
        let out: Vec<f32> = t_slice
            .iter()
            .enumerate()
            .map(|(idx, &val)| val + b_slice[idx % cols])
            .collect();
        Tensor::from_f32_slice(&out, tensor.shape().to_vec(), tensor.device().clone())
    }

    fn apply_per_head_rmsnorm(
        data: &mut [f32],
        weight: &[f32],
        num_tokens: usize,
        num_heads: usize,
        head_dim: usize,
        eps: f32,
    ) {
        let total_heads = num_tokens * num_heads;
        for h_idx in 0..total_heads {
            let offset = h_idx * head_dim;
            if offset + head_dim > data.len() {
                break;
            }
            let head_slice = &mut data[offset..offset + head_dim];
            let sum_sq: f32 = head_slice.iter().map(|&x| x * x).sum();
            let inv_rms = 1.0 / (sum_sq / (head_dim as f32) + eps).sqrt();
            for (d, val) in head_slice.iter_mut().enumerate() {
                let w = weight.get(d).copied().unwrap_or(1.0);
                *val = *val * inv_rms * w;
            }
        }
    }

    fn add_residual(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_slice = a.as_f32_slice()?;
        let b_slice = b.as_f32_slice()?;
        let out: Vec<f32> = a_slice
            .iter()
            .zip(b_slice.iter())
            .map(|(&x, &y)| x + y)
            .collect();
        Tensor::from_f32_slice(&out, a.shape().to_vec(), a.device().clone())
    }
}

