// HARNESS Native WGSL Kernel: Sparse MoE Top-K Softmax Router
// Computes numerically-stable Softmax over E routed experts (e.g., E=128 for Qwen3-Coder-30B-A3B),
// selects Top-K active experts per token (e.g., K=8), and normalizes routing weights so sum = 1.0.

struct MoeRouterUniform {
    num_tokens: u32,
    num_experts: u32,
    top_k: u32,
    norm_topk: u32,
}

@group(0) @binding(0) var<storage, read> router_logits: array<f32>;
@group(0) @binding(1) var<storage, read_write> topk_indices: array<u32>;
@group(0) @binding(2) var<storage, read_write> topk_weights: array<f32>;
@group(0) @binding(3) var<uniform> params: MoeRouterUniform;

var<workgroup> probs: array<f32, 256>;

@compute @workgroup_size(128, 1, 1)
fn moe_route_topk(
    @builtin(workgroup_id) wg_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let tok = wg_id.x;
    if (tok >= params.num_tokens) {
        return;
    }
    let tid = local_id.x;
    let num_e = min(params.num_experts, 256u);
    let base_in = tok * params.num_experts;

    // 1. Load logits into shared memory & find max for numerical stability
    if (tid < num_e) {
        probs[tid] = router_logits[base_in + tid];
    }
    if (tid + 128u < num_e) {
        probs[tid + 128u] = router_logits[base_in + tid + 128u];
    }
    workgroupBarrier();

    if (tid == 0u) {
        var max_val: f32 = -1e30;
        for (var e: u32 = 0u; e < num_e; e = e + 1u) {
            max_val = max(max_val, probs[e]);
        }

        var sum_exp: f32 = 0.0;
        for (var e: u32 = 0u; e < num_e; e = e + 1u) {
            let p = exp(probs[e] - max_val);
            probs[e] = p;
            sum_exp += p;
        }

        let inv_sum = 1.0 / max(sum_exp, 1e-12);
        for (var e: u32 = 0u; e < num_e; e = e + 1u) {
            probs[e] *= inv_sum;
        }

        // Select Top-K experts greedily and zero out selected slots
        let k_limit = min(params.top_k, num_e);
        let out_base = tok * params.top_k;
        var selected_sum: f32 = 0.0;

        for (var ki: u32 = 0u; ki < k_limit; ki = ki + 1u) {
            var best_idx: u32 = 0u;
            var best_prob: f32 = -1.0;
            for (var e: u32 = 0u; e < num_e; e = e + 1u) {
                if (probs[e] > best_prob) {
                    best_prob = probs[e];
                    best_idx = e;
                }
            }
            topk_indices[out_base + ki] = best_idx;
            topk_weights[out_base + ki] = best_prob;
            selected_sum += best_prob;
            probs[best_idx] = -1.0;
        }

        if (params.norm_topk != 0u && selected_sum > 1e-12) {
            let inv_sel = 1.0 / selected_sum;
            for (var ki: u32 = 0u; ki < k_limit; ki = ki + 1u) {
                topk_weights[out_base + ki] *= inv_sel;
            }
        }
    }
}
