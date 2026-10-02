// HARNESS High-Performance Tiled GEMM Compute Kernel (WGSL)
// Computes C = A * B
// A: [M, K], B: [K, N], C: [M, N]

struct Dimensions {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

@group(0) @binding(0) var<storage, read> matrix_a: array<f32>;
@group(0) @binding(1) var<storage, read> matrix_b: array<f32>;
@group(0) @binding(2) var<storage, read_write> matrix_c: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dimensions;

const TILE_SIZE: u32 = 16u;

var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

@compute @workgroup_size(16, 16, 1)
fn gemm_tiled(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) group_id: vec3<u32>
) {
    let row = global_id.y;
    let col = global_id.x;
    let local_row = local_id.y;
    let local_col = local_id.x;

    var acc: f32 = 0.0;
    let num_tiles = (dims.k + TILE_SIZE - 1u) / TILE_SIZE;

    for (var t: u32 = 0u; t < num_tiles; t = t + 1u) {
        // Cooperatively load tile from matrix A into shared workgroup memory
        let a_col = t * TILE_SIZE + local_col;
        if (row < dims.m && a_col < dims.k) {
            tile_a[local_row][local_col] = matrix_a[row * dims.k + a_col];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }

        // Cooperatively load tile from matrix B into shared workgroup memory
        let b_row = t * TILE_SIZE + local_row;
        if (b_row < dims.k && col < dims.n) {
            tile_b[local_row][local_col] = matrix_b[b_row * dims.n + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }

        workgroupBarrier();

        // Accumulate dot product across tile
        for (var i: u32 = 0u; i < TILE_SIZE; i = i + 1u) {
            acc = acc + tile_a[local_row][i] * tile_b[i][local_col];
        }

        workgroupBarrier();
    }

    if (row < dims.m && col < dims.n) {
        matrix_c[row * dims.n + col] = acc;
    }
}
