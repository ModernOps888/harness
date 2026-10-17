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

/// FlashAttention v3 tiled algorithm with online softmax and Grouped-Query Attention (GQA):
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
    let num_kv_heads = config.num_kv_heads.max(1);
    let head_dim = config.head_dim;

    if !num_q_heads.is_multiple_of(num_kv_heads) {
        return Err(harness_core::HarnessError::Attention(format!(
            "GQA requires num_q_heads ({}) to be a multiple of num_kv_heads ({})",
            num_q_heads, num_kv_heads
        )));
    }
    let gqa_ratio = num_q_heads / num_kv_heads;
    let tile_kv = config.tile_size_kv.max(1);

    let mut out = vec![0.0f32; seq_len_q * num_q_heads * head_dim];

    use rayon::prelude::*;

    // Parallelize query head processing across all available physical CPU cores
    let head_results: Vec<Vec<f32>> = (0..num_q_heads)
        .into_par_iter()
        .map(|qh| {
            let kv_h = qh / gqa_ratio;
            let mut head_out = vec![0.0f32; seq_len_q * head_dim];
            let mut acc = vec![0.0f32; head_dim];

            for qi in 0..seq_len_q {
                let q_offset = (qi * num_q_heads + qh) * head_dim;
                let q_vec = &q_slice[q_offset..q_offset + head_dim];

                // Online softmax accumulators: m = max logit, l = sum(exp(logit - m))
                let mut m_prev = -f32::INFINITY;
                let mut l_prev = 0.0f32;
                acc.fill(0.0);

                // Correct causal boundary for both prefill (seq_len_q == seq_len_kv)
                // and decode / speculative verification (seq_len_q < seq_len_kv)
                let max_k_pos = if config.causal {
                    (seq_len_kv.saturating_sub(seq_len_q) + qi + 1).min(seq_len_kv)
                } else {
                    seq_len_kv
                };

                let mut k_block_start = 0;
                while k_block_start < max_k_pos {
                    let k_block_end = (k_block_start + tile_kv).min(max_k_pos);
                    for ki in k_block_start..k_block_end {
                        let k_offset = (ki * num_kv_heads + kv_h) * head_dim;
                        let k_vec = &k_slice[k_offset..k_offset + head_dim];

                        // Dot product q * k with compiler auto-vectorization
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q_vec[d] * k_vec[d];
                        }
                        let s = dot * config.scale;

                        // Online rescaled softmax update step
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
                    k_block_start = k_block_end;
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

