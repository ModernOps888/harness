use rayon::prelude::*;

pub struct RotaryEmbedding {
    dim: usize,
    max_seq_len: usize,
    theta: f32,
    cos_table: Vec<f32>,
    sin_table: Vec<f32>,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        let half_dim = dim / 2;
        let mut cos_table = vec![0.0f32; max_seq_len * half_dim];
        let mut sin_table = vec![0.0f32; max_seq_len * half_dim];

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let freq = 1.0 / theta.powf((2 * i) as f32 / dim as f32);
                let angle = pos as f32 * freq;
                let idx = pos * half_dim + i;
                cos_table[idx] = angle.cos();
                sin_table[idx] = angle.sin();
            }
        }

        Self {
            dim,
            max_seq_len,
            theta,
            cos_table,
            sin_table,
        }
    }

    /// Apply Rotary Embedding to query or key tensor [num_tokens, num_heads, head_dim]
    pub fn apply(&self, x: &mut [f32], start_pos: usize, _num_tokens: usize, num_heads: usize) {
        let head_dim = self.dim;
        let half_dim = head_dim / 2;

        x.par_chunks_mut(num_heads * head_dim)
            .enumerate()
            .for_each(|(t, token_slice)| {
                let pos = start_pos + t;
                if pos >= self.max_seq_len {
                    return;
                }
                let table_offset = pos * half_dim;
                let cos_part = &self.cos_table[table_offset..table_offset + half_dim];
                let sin_part = &self.sin_table[table_offset..table_offset + half_dim];

                for h in 0..num_heads {
                    let head_slice = &mut token_slice[h * head_dim..(h + 1) * head_dim];
                    for i in 0..half_dim {
                        let x0 = head_slice[i];
                        let x1 = head_slice[i + half_dim];
                        let cos = cos_part[i];
                        let sin = sin_part[i];

                        // Standard 2D rotation
                        head_slice[i] = x0 * cos - x1 * sin;
                        head_slice[i + half_dim] = x0 * sin + x1 * cos;
                    }
                }
            });
    }
}
