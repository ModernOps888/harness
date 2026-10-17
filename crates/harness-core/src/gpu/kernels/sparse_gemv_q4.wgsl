// HARNESS Native WGSL Kernel: Dynamic Activation-Sparse In-Register INT4 GEMV
// Zero-barrier inner loop + Warp-uniform Q4Block broadcast + GDDR7 block-skipping
// when |x[k_idx]| <= dims.threshold (e.g. post-SwiGLU / ReLU2 activation sparsity).

struct Q4BlockGpu {
    scale: f32,
    min_val: f32,
    qs0: u32,
    qs1: u32,
    qs2: u32,
    qs3: u32,
}

struct SparseQ4GemvUniform {
    k: u32,
    n: u32,
    blocks_per_col: u32,
    threshold: f32,
}

@group(0) @binding(0) var<storage, read> vector_x: array<f32>;
@group(0) @binding(1) var<storage, read> blocks_w: array<Q4BlockGpu>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<uniform> dims: SparseQ4GemvUniform;

var<workgroup> shared_q4_acc: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn sparse_gemv_q4_decode(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let col = group_id.x;
    let tid = local_id.x;
    let warp_id = tid / 32u;       // 0..7 (8 warps per workgroup -> 8 Q4 blocks = 256 weights per tile)
    let lane = tid % 32u;          // 0..31 (1 weight per lane inside the 32-weight Q4 block)
    let chunk = lane / 8u;         // 0..3 (selects qs0, qs1, qs2, or qs3)
    let shift = (lane % 8u) * 4u;  // Bit shift (0..28) for this lane's 4-bit nibble

    var thread_sum: f32 = 0.0;
    let num_tiles = (dims.blocks_per_col + 7u) / 8u;
    let col_block_base = col * dims.blocks_per_col;

    if (col < dims.n) {
        // Zero-barrier inner loop: each lane reads its own activation x_val directly from L1 cache
        // and only issues a GDDR7 Q4Block load when |x_val| > threshold.
        for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
            let b = t * 8u + warp_id;
            let k_idx = t * 256u + tid;
            if (b < dims.blocks_per_col && k_idx < dims.k) {
                let x_val = vector_x[k_idx];
                if (abs(x_val) > dims.threshold) {
                    let block = blocks_w[col_block_base + b];
                    var word: u32 = block.qs0;
                    if (chunk == 1u) {
                        word = block.qs1;
                    } else if (chunk == 2u) {
                        word = block.qs2;
                    } else if (chunk == 3u) {
                        word = block.qs3;
                    }
                    let nibble = f32((word >> shift) & 0x0Fu);
                    let weight = fma(nibble, block.scale, block.min_val);
                    thread_sum = fma(x_val, weight, thread_sum);
                }
            }
        }
    }

    shared_q4_acc[tid] = thread_sum;
    workgroupBarrier();

    // Parallel tree reduction across 256 workgroup threads
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
