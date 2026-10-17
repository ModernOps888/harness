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

var<workgroup> shared_x: array<f32, 256>;
var<workgroup> shared_acc: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn gemv_decode(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let col = group_id.x;
    let tid = local_id.x;
    var sum: f32 = 0.0;

    let num_tiles = (dims.k + 255u) / 256u;

    // Cooperatively load vector_x tiles into shared workgroup memory before multiplying
    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let k = t * 256u + tid;
        if (k < dims.k) {
            shared_x[tid] = vector_x[k];
        } else {
            shared_x[tid] = 0.0;
        }
        workgroupBarrier();

        if (col < dims.n && k < dims.k) {
            let w_val = matrix_w[k * dims.n + col];
            sum = sum + shared_x[tid] * w_val;
        }
        workgroupBarrier();
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

    if (tid == 0u && col < dims.n) {
        output_y[col] = shared_acc[0];
    }
}
