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

var<workgroup> shared_q4_acc: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn gemv_q4_decode(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let col = group_id.x;
    if (col >= dims.n) {
        return;
    }

    let tid = local_id.x;
    var thread_sum: f32 = 0.0;

    // Each thread processes a stride of 32-element Q4 blocks down column 'col'
    for (var b = tid; b < dims.blocks_per_col; b = b + 256u) {
        let block_idx = col * dims.blocks_per_col + b;
        let block = blocks_w[block_idx];
        let scale = block.scale;
        let min_val = block.min_val;
        let x_offset = b * 32u;

        // Unpack qs0 (8 nibbles: 0..7)
        let w0 = block.qs0;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let nibble = f32((w0 >> (i * 4u)) & 0x0Fu);
            let weight = nibble * scale + min_val;
            thread_sum = thread_sum + vector_x[x_offset + i] * weight;
        }

        // Unpack qs1 (8 nibbles: 8..15)
        let w1 = block.qs1;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let nibble = f32((w1 >> (i * 4u)) & 0x0Fu);
            let weight = nibble * scale + min_val;
            thread_sum = thread_sum + vector_x[x_offset + 8u + i] * weight;
        }

        // Unpack qs2 (8 nibbles: 16..23)
        let w2 = block.qs2;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let nibble = f32((w2 >> (i * 4u)) & 0x0Fu);
            let weight = nibble * scale + min_val;
            thread_sum = thread_sum + vector_x[x_offset + 16u + i] * weight;
        }

        // Unpack qs3 (8 nibbles: 24..31)
        let w3 = block.qs3;
        for (var i = 0u; i < 8u; i = i + 1u) {
            let nibble = f32((w3 >> (i * 4u)) & 0x0Fu);
            let weight = nibble * scale + min_val;
            thread_sum = thread_sum + vector_x[x_offset + 24u + i] * weight;
        }
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

    if (tid == 0u) {
        output_y[col] = shared_q4_acc[0];
    }
}
