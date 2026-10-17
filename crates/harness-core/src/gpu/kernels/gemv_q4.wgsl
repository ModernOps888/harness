// HARNESS Native Quantized Q4 Vector-Matrix Multiply (GEMV) Kernel (WGSL)
// Directly reads 4-bit quantized weights from VRAM and dequantizes in-register
// Eliminates 81.25% of VRAM memory bandwidth pressure (5.33x bandwidth multiplier)

struct Q4BlockGpu {
    scale: f32,
    min_val: f32,
    qs0: u32, // 8 nibbles (weights 0..7)
    qs1: u32, // 8 nibbles (weights 8..15)
    qs2: u32, // 8 nibbles (weights 16..23)
    qs3: u32, // 8 nibbles (weights 24..31)
}

struct Q4GemvDims {
    k: u32,          // Total elements in vector x
    n: u32,          // Output columns
    blocks_per_col: u32, // k / 32
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> vector_x: array<f32>;
@group(0) @binding(1) var<storage, read> blocks_w: array<Q4BlockGpu>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Q4GemvDims;

var<workgroup> shared_x: array<f32, 256>;
var<workgroup> shared_q4_acc: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn gemv_q4_decode(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let col = group_id.x;
    let tid = local_id.x;
    let warp_id = tid / 32u;       // 0..7 (8 warps per workgroup -> 8 Q4 blocks = 256 weights per tile)
    let lane = tid % 32u;          // 0..31 (1 weight per lane inside the 32-weight Q4 block)
    let chunk = lane / 8u;         // 0..3 (selects qs0, qs1, qs2, or qs3)
    let shift = (lane % 8u) * 4u;  // Bit shift (0..28) for this lane's 4-bit nibble

    var thread_sum: f32 = 0.0;
    let num_tiles = (dims.blocks_per_col + 7u) / 8u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let b_base = t * 8u;
        let k_idx = b_base * 32u + tid;

        // 100% coalesced unit-stride load of 256 elements of vector_x into shared memory
        if (k_idx < dims.k) {
            shared_x[tid] = vector_x[k_idx];
        } else {
            shared_x[tid] = 0.0;
        }
        workgroupBarrier();

        let b = b_base + warp_id;
        if (col < dims.n && b < dims.blocks_per_col && k_idx < dims.k) {
            let block_idx = col * dims.blocks_per_col + b;
            let block = blocks_w[block_idx];

            var word: u32 = block.qs0;
            if (chunk == 1u) {
                word = block.qs1;
            } else if (chunk == 2u) {
                word = block.qs2;
            } else if (chunk == 3u) {
                word = block.qs3;
            }

            let nibble = f32((word >> shift) & 0x0Fu);
            let weight = nibble * block.scale + block.min_val;
            thread_sum = thread_sum + shared_x[tid] * weight;
        }
        workgroupBarrier();
    }

    shared_q4_acc[tid] = thread_sum;
    workgroupBarrier();

    // Parallel reduction across 256 workgroup threads
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_q4_acc[tid] = shared_q4_acc[tid] + shared_q4_acc[tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u && col < dims.n) {
        output_y[col] = shared_q4_acc[0];
    }
}
