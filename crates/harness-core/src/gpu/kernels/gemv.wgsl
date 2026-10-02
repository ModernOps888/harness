// HARNESS Ultra-Fast Vector-Matrix Multiply (GEMV) Kernel for M=1 Autoregressive Token Decoding
// Computes y = x * W
// x: [1, K], W: [K, N], y: [1, N]

struct GemvDims {
    k: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> vector_x: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_w: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<uniform> dims: GemvDims;

var<workgroup> shared_acc: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn gemv_decode(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let col = group_id.x;
    if (col >= dims.n) {
        return;
    }

    let tid = local_id.x;
    var sum: f32 = 0.0;

    // Strided accumulation across K dimension
    for (var k = tid; k < dims.k; k = k + 256u) {
        let x_val = vector_x[k];
        let w_val = matrix_w[k * dims.n + col];
        sum = sum + x_val * w_val;
    }

    shared_acc[tid] = sum;
    workgroupBarrier();

    // Parallel reduction in workgroup shared memory
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_acc[tid] = shared_acc[tid] + shared_acc[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        output_y[col] = shared_acc[0];
    }
}
