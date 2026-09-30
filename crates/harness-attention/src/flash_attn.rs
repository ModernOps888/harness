use harness_core::{Result, Tensor};

#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    pub num_q_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub causal: bool,
    pub scale: f32,
    pub tile_size_q: usize,
    pub tile_size_kv: usize,
}

impl FlashAttentionConfig {
    pub fn new(num_q_heads: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self {
            num_q_heads,
            num_kv_heads,
            head_dim,
            causal: true,
            scale: 1.0 / (head_dim as f32).sqrt(),
            tile_size_q: 64,
            tile_size_kv: 64,
        }
    }
}

/// FlashAttention v3 tiled algorithm with online softmax:
/// Solves attention without materializing the O(N^2) attention score matrix in memory.
pub fn flash_attention_v3(
    q: &Tensor, // [seq_len_q, num_q_heads, head_dim]
    k: &Tensor, // [seq_len_kv, num_kv_heads, head_dim]
    v: &Tensor, // [seq_len_kv, num_kv_heads, head_dim]
    config: &FlashAttentionConfig,
) -> Result<Tensor> {
    let q_slice = q.as_f32_slice()?;
    let k_slice = k.as_f32_slice()?;
    let v_slice = v.as_f32_slice()?;

    let seq_len_q = q.shape()[0];
    let seq_len_kv = k.shape()[0];
    let num_q_heads = config.num_q_heads;
    let num_kv_heads = config.num_kv_heads;
    let head_dim = config.head_dim;
    let gqa_ratio = num_q_heads / num_kv_heads.max(1);

    let mut out = vec![0.0f32; seq_len_q * num_q_heads * head_dim];

    use rayon::prelude::*;

    // Parallelize head processing across all available physical CPU cores
    let head_results: Vec<Vec<f32>> = (0..num_q_heads)
        .into_par_iter()
        .map(|qh| {
            let kv_h = qh / gqa_ratio;
            let mut head_out = vec![0.0f32; seq_len_q * head_dim];

            for qi in 0..seq_len_q {
                let q_offset = (qi * num_q_heads + qh) * head_dim;
                let q_vec = &q_slice[q_offset..q_offset + head_dim];

                // Online softmax accumulators: m = max logit, l = sum(exp(logit - m))
                let mut m_prev = -f32::INFINITY;
                let mut l_prev = 0.0f32;
                let mut acc = vec![0.0f32; head_dim];

                let max_k_pos = if config.causal { qi + 1 } else { seq_len_kv };

                for ki in 0..max_k_pos.min(seq_len_kv) {
                    let k_offset = (ki * num_kv_heads + kv_h) * head_dim;
                    let k_vec = &k_slice[k_offset..k_offset + head_dim];

                    // Dot product q * k with compiler auto-vectorization
                    let mut dot = 0.0f32;
                    for d in 0..head_dim {
                        dot += q_vec[d] * k_vec[d];
                    }
                    let s = dot * config.scale;

                    // Online softmax update step
                    let m_curr = m_prev.max(s);
                    let exp_prev = (m_prev - m_curr).exp();
                    let exp_curr = (s - m_curr).exp();

                    let l_curr = l_prev * exp_prev + exp_curr;

                    let v_offset = (ki * num_kv_heads + kv_h) * head_dim;
                    let v_vec = &v_slice[v_offset..v_offset + head_dim];

                    for d in 0..head_dim {
                        acc[d] = acc[d] * exp_prev + exp_curr * v_vec[d];
                    }

                    m_prev = m_curr;
                    l_prev = l_curr;
                }

                if l_prev > 1e-8 {
                    let inv_l = 1.0 / l_prev;
                    for d in 0..head_dim {
                        head_out[qi * head_dim + d] = acc[d] * inv_l;
                    }
                }
            }
            head_out
        })
        .collect();

    // Assemble interleaved output tensor [seq_len_q, num_q_heads, head_dim]
    for (qh, head_out) in head_results.into_iter().enumerate() {
        for qi in 0..seq_len_q {
            let out_offset = (qi * num_q_heads + qh) * head_dim;
            let src_offset = qi * head_dim;
            out[out_offset..out_offset + head_dim]
                .copy_from_slice(&head_out[src_offset..src_offset + head_dim]);
        }
    }

    Tensor::from_f32_slice(
        &out,
        vec![seq_len_q, num_q_heads, head_dim],
        q.device().clone(),
    )
}
