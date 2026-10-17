// HARNESS Batched Quantized Q4 Micro-GEMM Kernel (WGSL)
// Designed for M = K_draft + 1 Speculative Decoding Verification (e.g. M = 5..8 tokens)
// Reads 4-bit quantized weights from VRAM ONLY ONCE and multiplies against all M token rows simultaneously in registers!

struct Q4BlockGpu {
    scale: f32,
    min_val: f32,
    qs0: u32, // 8 nibbles (weights 0..7)
    qs1: u32, // 8 nibbles (weights 8..15)
    qs2: u32, // 8 nibbles (weights 16..23)
    qs3: u32, // 8 nibbles (weights 24..31)
}

struct Q4GemmBatchedDims {
    m: u32,              // Number of tokens in batch (M = K_draft + 1)
    k: u32,              // Input hidden dimension K
    n: u32,              // Output columns N
    blocks_per_col: u32, // ceil(K / 32)
}

@group(0) @binding(0) var<storage, read> matrix_x: array<f32>;
@group(0) @binding(1) var<storage, read> blocks_w: array<Q4BlockGpu>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<uniform> dims: Q4GemmBatchedDims;

var<workgroup> shared_q4_batch_acc: array<array<f32, 256>, 8>;

@compute @workgroup_size(256, 1, 1)
fn gemm_q4_batched(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let col = group_id.x;
    let m_base = group_id.y * 8u;
    let tid = local_id.x;

    let warp_id = tid / 32u;       // 0..7 (8 warps per workgroup -> 8 Q4 blocks = 256 weights per step)
    let lane = tid % 32u;          // 0..31 (1 weight per lane inside the 32-weight Q4 block)
    let chunk = lane / 8u;         // 0..3 (selects qs0, qs1, qs2, or qs3)
    let shift = (lane % 8u) * 4u;  // Bit shift (0..28) for this lane's 4-bit nibble

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    var acc4: f32 = 0.0;
    var acc5: f32 = 0.0;
    var acc6: f32 = 0.0;
    var acc7: f32 = 0.0;

    let num_tiles = (dims.blocks_per_col + 7u) / 8u;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        let b_base = t * 8u;
        let b = b_base + warp_id;
        let k_idx = b_base * 32u + tid;

        if (col < dims.n && b < dims.blocks_per_col && k_idx < dims.k) {
            // Read and dequantize the 4-bit weight strictly ONCE from VRAM into register
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

            // Multiply single in-register weight across all M token rows simultaneously (100% coalesced reads)
            if (m_base + 0u < dims.m) { acc0 = acc0 + matrix_x[(m_base + 0u) * dims.k + k_idx] * weight; }
            if (m_base + 1u < dims.m) { acc1 = acc1 + matrix_x[(m_base + 1u) * dims.k + k_idx] * weight; }
            if (m_base + 2u < dims.m) { acc2 = acc2 + matrix_x[(m_base + 2u) * dims.k + k_idx] * weight; }
            if (m_base + 3u < dims.m) { acc3 = acc3 + matrix_x[(m_base + 3u) * dims.k + k_idx] * weight; }
            if (m_base + 4u < dims.m) { acc4 = acc4 + matrix_x[(m_base + 4u) * dims.k + k_idx] * weight; }
            if (m_base + 5u < dims.m) { acc5 = acc5 + matrix_x[(m_base + 5u) * dims.k + k_idx] * weight; }
            if (m_base + 6u < dims.m) { acc6 = acc6 + matrix_x[(m_base + 6u) * dims.k + k_idx] * weight; }
            if (m_base + 7u < dims.m) { acc7 = acc7 + matrix_x[(m_base + 7u) * dims.k + k_idx] * weight; }
        }
    }

    shared_q4_batch_acc[0][tid] = acc0;
    shared_q4_batch_acc[1][tid] = acc1;
    shared_q4_batch_acc[2][tid] = acc2;
    shared_q4_batch_acc[3][tid] = acc3;
    shared_q4_batch_acc[4][tid] = acc4;
    shared_q4_batch_acc[5][tid] = acc5;
    shared_q4_batch_acc[6][tid] = acc6;
    shared_q4_batch_acc[7][tid] = acc7;
    workgroupBarrier();

    // Parallel reduction across 256 workgroup threads for all active batch rows
    for (var stride = 128u; stride > 0u; stride = stride >> 1u) {
        if (tid < stride) {
            shared_q4_batch_acc[0][tid] = shared_q4_batch_acc[0][tid] + shared_q4_batch_acc[0][tid + stride];
            shared_q4_batch_acc[1][tid] = shared_q4_batch_acc[1][tid] + shared_q4_batch_acc[1][tid + stride];
            shared_q4_batch_acc[2][tid] = shared_q4_batch_acc[2][tid] + shared_q4_batch_acc[2][tid + stride];
            shared_q4_batch_acc[3][tid] = shared_q4_batch_acc[3][tid] + shared_q4_batch_acc[3][tid + stride];
            shared_q4_batch_acc[4][tid] = shared_q4_batch_acc[4][tid] + shared_q4_batch_acc[4][tid + stride];
            shared_q4_batch_acc[5][tid] = shared_q4_batch_acc[5][tid] + shared_q4_batch_acc[5][tid + stride];
            shared_q4_batch_acc[6][tid] = shared_q4_batch_acc[6][tid] + shared_q4_batch_acc[6][tid + stride];
            shared_q4_batch_acc[7][tid] = shared_q4_batch_acc[7][tid] + shared_q4_batch_acc[7][tid + stride];
        }
        workgroupBarrier();
    }

    if (tid == 0u && col < dims.n) {
        for (var r: u32 = 0u; r < 8u; r = r + 1u) {
            let row = m_base + r;
            if (row < dims.m) {
                output_y[row * dims.n + col] = shared_q4_batch_acc[r][0];
            }
        }
    }
}
