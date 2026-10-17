// HARNESS Fused Grouped-Query Attention (GQA) FlashAttention Kernel (WGSL)
// Natively supports Qwen2.5-Coder-32B's 5:1 GQA head ratio (num_q_heads=40, num_kv_heads=8, head_dim=128)
// Streams each K and V head from VRAM ONLY ONCE and shares it across all GQA query heads in registers
// with online softmax and causal / speculative batch verification masking.

struct FlashAttnGqaParams {
    seq_len_q: u32,
    seq_len_kv: u32,
    num_q_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    gqa_ratio: u32,
    causal: u32,
    scale: f32,
}

@group(0) @binding(0) var<storage, read> query_q: array<f32>;
@group(0) @binding(1) var<storage, read> key_k: array<f32>;
@group(0) @binding(2) var<storage, read> value_v: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_o: array<f32>;
@group(0) @binding(4) var<uniform> params: FlashAttnGqaParams;

var<workgroup> shared_dots: array<array<f32, 128>, 8>;

@compute @workgroup_size(128, 1, 1)
fn flash_attn_gqa(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let g_chunks = (params.gqa_ratio + 7u) / 8u;
    let kv_h = group_id.x / g_chunks;
    let g_base = (group_id.x % g_chunks) * 8u;
    let qi = group_id.y;
    let tid = local_id.x;

    if (kv_h >= params.num_kv_heads || qi >= params.seq_len_q) {
        return;
    }

    // Load all GQA query heads sharing this kv_h into registers for channel `tid`
    var q_reg: array<f32, 8>;
    var m_prev: array<f32, 8>;
    var l_prev: array<f32, 8>;
    var acc: array<f32, 8>;

    for (var g: u32 = 0u; g < 8u; g = g + 1u) {
        let q_group_idx = g_base + g;
        m_prev[g] = -1e30;
        l_prev[g] = 0.0;
        acc[g] = 0.0;
        if (q_group_idx < params.gqa_ratio && tid < params.head_dim) {
            let qh = kv_h * params.gqa_ratio + q_group_idx;
            let q_idx = (qi * params.num_q_heads + qh) * params.head_dim + tid;
            q_reg[g] = query_q[q_idx];
        } else {
            q_reg[g] = 0.0;
        }
    }

    // Causal / speculative verification mask:
    // If seq_len_kv >= seq_len_q (e.g., verifying M = K+1 draft tokens appended to a prefix cache),
    // query `qi` attends up to `(seq_len_kv - seq_len_q) + qi + 1`.
    var max_k_pos = params.seq_len_kv;
    if (params.causal != 0u) {
        let past_len = select(0u, params.seq_len_kv - params.seq_len_q, params.seq_len_kv >= params.seq_len_q);
        max_k_pos = min(params.seq_len_kv, past_len + qi + 1u);
    }

    for (var ki: u32 = 0u; ki < max_k_pos; ki = ki + 1u) {
        let kv_base = (ki * params.num_kv_heads + kv_h) * params.head_dim;

        // Coalesced single read of K and V for this KV head
        var k_val: f32 = 0.0;
        var v_val: f32 = 0.0;
        if (tid < params.head_dim) {
            k_val = key_k[kv_base + tid];
            v_val = value_v[kv_base + tid];
        }

        for (var g: u32 = 0u; g < 8u; g = g + 1u) {
            shared_dots[g][tid] = q_reg[g] * k_val;
        }
        workgroupBarrier();

        // Parallel reduction across 128 head_dim channels for all GQA heads in lockstep
        for (var stride = 64u; stride > 0u; stride = stride >> 1u) {
            if (tid < stride) {
                for (var g: u32 = 0u; g < 8u; g = g + 1u) {
                    shared_dots[g][tid] = shared_dots[g][tid] + shared_dots[g][tid + stride];
                }
            }
            workgroupBarrier();
        }

        // Online softmax update and V accumulation in registers across all GQA query heads
        for (var g: u32 = 0u; g < 8u; g = g + 1u) {
            if (g_base + g < params.gqa_ratio) {
                let s = shared_dots[g][0] * params.scale;
                let m_curr = max(m_prev[g], s);
                let exp_prev = exp(m_prev[g] - m_curr);
                let exp_curr = exp(s - m_curr);
                l_prev[g] = l_prev[g] * exp_prev + exp_curr;
                acc[g] = acc[g] * exp_prev + exp_curr * v_val;
                m_prev[g] = m_curr;
            }
        }
        workgroupBarrier();
    }

    // Write normalized output for all GQA query heads
    if (tid < params.head_dim) {
        for (var g: u32 = 0u; g < 8u; g = g + 1u) {
            let q_group_idx = g_base + g;
            if (q_group_idx < params.gqa_ratio) {
                let qh = kv_h * params.gqa_ratio + q_group_idx;
                let out_idx = (qi * params.num_q_heads + qh) * params.head_dim + tid;
                if (l_prev[g] > 1e-8) {
                    output_o[out_idx] = acc[g] / l_prev[g];
                } else {
                    output_o[out_idx] = 0.0;
                }
            }
        }
    }
}
