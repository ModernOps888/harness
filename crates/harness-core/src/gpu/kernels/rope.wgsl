// HARNESS Rotary Positional Embeddings (RoPE) Kernel (WGSL)
// Applies 2D rotations to pairs of channels along head dimension

struct RoPEParams {
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    start_pos: u32,
    freq_base: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

@group(0) @binding(0) var<storage, read> input_t: array<f32>;
@group(0) @binding(1) var<storage, read_write> output_t: array<f32>;
@group(0) @binding(2) var<uniform> params: RoPEParams;

@compute @workgroup_size(256, 1, 1)
fn rope_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let pair_idx = global_id.x;
    let pairs_per_head = params.head_dim / 2u;
    let total_pairs = params.num_tokens * params.num_heads * pairs_per_head;

    if (pair_idx >= total_pairs) {
        return;
    }

    let i = pair_idx % pairs_per_head;
    let head_token = pair_idx / pairs_per_head;
    let head_idx = head_token % params.num_heads;
    let token_idx = head_token / params.num_heads;
    let pos = f32(params.start_pos + token_idx);

    // Compute rotation angle theta
    let exponent = 2.0 * f32(i) / f32(params.head_dim);
    let freq = 1.0 / pow(params.freq_base, exponent);
    let theta = pos * freq;
    let cos_t = cos(theta);
    let sin_t = sin(theta);

    let base_offset = (token_idx * params.num_heads + head_idx) * params.head_dim;
    let idx0 = base_offset + 2u * i;
    let idx1 = base_offset + 2u * i + 1u;

    let x0 = input_t[idx0];
    let x1 = input_t[idx1];

    output_t[idx0] = x0 * cos_t - x1 * sin_t;
    output_t[idx1] = x0 * sin_t + x1 * cos_t;
}
