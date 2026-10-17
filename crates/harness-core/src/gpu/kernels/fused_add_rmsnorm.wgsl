// HARNESS Fused Residual-Add + RMSNorm Compute Kernel (WGSL)
// Computes h = x + residual; y = (h / sqrt(mean(h^2) + eps)) * weight in a single GPU dispatch

struct NormParams {
    num_tokens: u32,
    hidden_dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> residual: array<f32>;
@group(0) @binding(2) var<storage, read> weight: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(4) var<uniform> params: NormParams;

var<workgroup> shared_sq_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn fused_add_rms_norm(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let token_idx = group_id.x;
    let tid = local_id.x;
    let row_offset = token_idx * params.hidden_dim;

    // 1. Thread-local fused residual addition + sum of squares
    var local_sum: f32 = 0.0;
    if (token_idx < params.num_tokens) {
        for (var i = tid; i < params.hidden_dim; i = i + 256u) {
            let h_val = input_x[row_offset + i] + residual[row_offset + i];
            local_sum = local_sum + h_val * h_val;
        }
    }
    shared_sq_sum[tid] = local_sum;
    workgroupBarrier();

    // 2. Parallel reduction in workgroup shared memory
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_sq_sum[tid] = shared_sq_sum[tid] + shared_sq_sum[tid + stride];
        }
        workgroupBarrier();
    }

    // 3. Compute RMS scale factor
    let mean_sq = shared_sq_sum[0] / f32(params.hidden_dim);
    let scale = 1.0 / sqrt(mean_sq + params.eps);
    workgroupBarrier();

    // 4. Normalize fused residual and apply affine weight
    if (token_idx < params.num_tokens) {
        for (var i = tid; i < params.hidden_dim; i = i + 256u) {
            let h_val = input_x[row_offset + i] + residual[row_offset + i];
            let w_val = weight[i];
            output_y[row_offset + i] = h_val * scale * w_val;
        }
    }
}
