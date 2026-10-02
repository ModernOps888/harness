// HARNESS Parallel Reduction RMSNorm Kernel (WGSL)
// Computes y = (x / sqrt(mean(x^2) + eps)) * weight

struct NormParams {
    num_tokens: u32,
    hidden_dim: u32,
    eps: f32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> weight: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<uniform> params: NormParams;

var<workgroup> shared_sq_sum: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn rms_norm(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let token_idx = group_id.x;
    if (token_idx >= params.num_tokens) {
        return;
    }

    let tid = local_id.x;
    let row_offset = token_idx * params.hidden_dim;

    // 1. Thread-local sum of squares
    var local_sum: f32 = 0.0;
    for (var i = tid; i < params.hidden_dim; i = i + 256u) {
        let val = input_x[row_offset + i];
        local_sum = local_sum + val * val;
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

    // 3. Compute scale factor
    let mean_sq = shared_sq_sum[0] / f32(params.hidden_dim);
    let scale = 1.0 / sqrt(mean_sq + params.eps);

    workgroupBarrier();

    // 4. Normalize and apply affine weight
    for (var i = tid; i < params.hidden_dim; i = i + 256u) {
        let x_val = input_x[row_offset + i];
        let w_val = weight[i];
        output_y[row_offset + i] = x_val * scale * w_val;
    }
}
