// HARNESS Fused SwiGLU Activation Kernel (WGSL)
// Computes y = (x * (1.0 / (1.0 + exp(-x)))) * gate

struct ActParams {
    num_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input_x: array<f32>;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_y: array<f32>;
@group(0) @binding(3) var<uniform> params: ActParams;

@compute @workgroup_size(256, 1, 1)
fn swiglu_forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= params.num_elements) {
        return;
    }

    let x = input_x[idx];
    let g = gate[idx];

    // Swish activation: x * sigmoid(x)
    let sigmoid_x = 1.0 / (1.0 + exp(-x));
    let swish = x * sigmoid_x;

    output_y[idx] = swish * g;
}
