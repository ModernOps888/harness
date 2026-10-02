use crate::gpu::GpuContext;
use crate::{HarnessError, Result};
use std::sync::Arc;
use wgpu::util::DeviceExt;

/// Native GPU Tensor stored directly in device VRAM
#[derive(Clone)]
pub struct GpuTensor {
    pub buffer: Arc<wgpu::Buffer>,
    pub shape: Vec<usize>,
    pub numel: usize,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemmDimensions {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct NormUniform {
    num_tokens: u32,
    hidden_dim: u32,
    eps: f32,
    _pad: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ActUniform {
    num_elements: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RoPEUniform {
    num_tokens: u32,
    num_heads: u32,
    head_dim: u32,
    start_pos: u32,
    freq_base: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GemvUniform {
    k: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Q4BlockGpu {
    pub scale: f32,
    pub min_val: f32,
    pub qs0: u32,
    pub qs1: u32,
    pub qs2: u32,
    pub qs3: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Q4GemvUniform {
    k: u32,
    n: u32,
    blocks_per_col: u32,
    _pad: u32,
}

/// Native GPU 4-Bit Quantized Weight Matrix stored directly in device VRAM
#[derive(Clone)]
pub struct GpuQ4Tensor {
    pub buffer: Arc<wgpu::Buffer>,
    pub k: usize,
    pub n: usize,
    pub num_blocks: usize,
}

impl GpuQ4Tensor {
    /// Create GpuQ4Tensor from pre-quantized GPU blocks
    pub fn from_blocks(ctx: &GpuContext, blocks: &[Q4BlockGpu], k: usize, n: usize) -> Result<Self> {
        let blocks_per_col = k.div_ceil(32);
        let expected_blocks = n * blocks_per_col;
        if blocks.len() != expected_blocks {
            return Err(HarnessError::InvalidShape(format!(
                "Q4 blocks length {} does not match expected {} for [K={}, N={}]",
                blocks.len(),
                expected_blocks,
                k,
                n
            )));
        }

        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuQ4Tensor VRAM Buffer"),
            contents: bytemuck::cast_slice(blocks),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            buffer: Arc::new(buffer),
            k,
            n,
            num_blocks: blocks.len(),
        })
    }

    /// Quantize a row-major F32 matrix of shape [K, N] into column-major GPU Q4 blocks
    /// and upload directly to device VRAM.
    pub fn from_f32_matrix(ctx: &GpuContext, data: &[f32], k: usize, n: usize) -> Result<Self> {
        if data.len() != k * n {
            return Err(HarnessError::InvalidShape(format!(
                "Data length {} does not match K*N = {}",
                data.len(),
                k * n
            )));
        }

        let blocks_per_col = k.div_ceil(32);
        let mut blocks = Vec::with_capacity(n * blocks_per_col);

        for col in 0..n {
            for b in 0..blocks_per_col {
                let start_k = b * 32;
                let end_k = (start_k + 32).min(k);

                let mut min_val = f32::MAX;
                let mut max_val = f32::MIN;

                for ki in start_k..end_k {
                    let val = data[ki * n + col];
                    if val < min_val {
                        min_val = val;
                    }
                    if val > max_val {
                        max_val = val;
                    }
                }

                if start_k >= end_k {
                    min_val = 0.0;
                    max_val = 0.0;
                }

                let diff = max_val - min_val;
                let scale = if diff > 1e-8 { diff / 15.0 } else { 1.0 };
                let inv_scale = 1.0 / scale;

                let mut qs = [0u32; 4];

                for (chunk_idx, q_chunk) in qs.iter_mut().enumerate() {
                    let mut packed_u32 = 0u32;
                    for nibble_idx in 0..8 {
                        let ki = start_k + chunk_idx * 8 + nibble_idx;
                        let q = if ki < end_k {
                            let val = data[ki * n + col];
                            (((val - min_val) * inv_scale).round() as u32).min(15)
                        } else {
                            0
                        };
                        packed_u32 |= q << (nibble_idx * 4);
                    }
                    *q_chunk = packed_u32;
                }

                blocks.push(Q4BlockGpu {
                    scale,
                    min_val,
                    qs0: qs[0],
                    qs1: qs[1],
                    qs2: qs[2],
                    qs3: qs[3],
                });
            }
        }

        Self::from_blocks(ctx, &blocks, k, n)
    }

    /// Size in bytes allocated in GPU VRAM
    pub fn memory_size_bytes(&self) -> usize {
        self.num_blocks * std::mem::size_of::<Q4BlockGpu>()
    }
}

impl GpuTensor {
    /// Upload CPU float slice to a high-speed GPU storage buffer in VRAM
    pub fn from_f32_slice(ctx: &GpuContext, data: &[f32], shape: Vec<usize>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        if data.len() != numel {
            return Err(HarnessError::InvalidShape(format!(
                "Data length {} does not match shape {:?} (expected {})",
                data.len(),
                shape,
                numel
            )));
        }

        let buffer = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GpuTensor VRAM Buffer"),
            contents: bytemuck::cast_slice(data),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape,
            numel,
        })
    }

    /// Allocate uninitialized or zeroed GPU VRAM tensor
    pub fn empty(ctx: &GpuContext, shape: Vec<usize>) -> Result<Self> {
        let numel: usize = shape.iter().product();
        let size_bytes = (numel * std::mem::size_of::<f32>()) as u64;

        let buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor Empty VRAM Buffer"),
            size: size_bytes.max(4),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer: Arc::new(buffer),
            shape,
            numel,
        })
    }

    /// Readback GPU VRAM buffer into CPU vector
    pub fn to_vec(&self, ctx: &GpuContext) -> Result<Vec<f32>> {
        let size_bytes = (self.numel * std::mem::size_of::<f32>()) as u64;

        let staging_buffer = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("GpuTensor Staging Read Buffer"),
            size: size_bytes.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GpuTensor Readback Encoder"),
        });

        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, size_bytes);
        ctx.queue.submit(Some(encoder.finish()));

        let buffer_slice = staging_buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::channel();
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            sender.send(result).ok();
        });

        ctx.device.poll(wgpu::Maintain::Wait);

        receiver
            .recv()
            .map_err(|e| HarnessError::Internal(format!("Buffer map channel error: {}", e)))?
            .map_err(|e| HarnessError::Internal(format!("Buffer mapping failed: {:?}", e)))?;

        let data = buffer_slice.get_mapped_range();
        let result: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        staging_buffer.unmap();

        Ok(result)
    }

    /// Native High-Speed GPU Matrix Multiplication: C = self * other
    pub fn matmul(&self, ctx: &GpuContext, other: &Self) -> Result<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(HarnessError::InvalidShape(
                "GPU GEMM requires 2D matrices [M, K] and [K, N]".into(),
            ));
        }

        let m = self.shape[0];
        let k1 = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];

        if k1 != k2 {
            return Err(HarnessError::ShapeMismatch {
                expected: vec![m, k1],
                found: vec![k2, n],
            });
        }

        let out = Self::empty(ctx, vec![m, n])?;

        let dims = GemmDimensions {
            m: m as u32,
            k: k1 as u32,
            n: n as u32,
            _pad: 0,
        };

        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GEMM Dims Uniform"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GEMM BindGroup"),
            layout: &ctx.gemm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: other.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GEMM Command Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GEMM Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.gemm_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let gx = (n as u32).div_ceil(16);
            let gy = (m as u32).div_ceil(16);
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        Ok(out)
    }

    /// Native Fused GPU RMSNorm
    pub fn rms_norm(&self, ctx: &GpuContext, weight: &Self, eps: f32) -> Result<Self> {
        let hidden_dim = *self.shape.last().unwrap_or(&1);
        let num_tokens = self.numel / hidden_dim;

        if weight.numel < hidden_dim {
            return Err(HarnessError::ShapeMismatch {
                expected: vec![hidden_dim],
                found: weight.shape.clone(),
            });
        }

        let out = Self::empty(ctx, self.shape.clone())?;

        let params = NormUniform {
            num_tokens: num_tokens as u32,
            hidden_dim: hidden_dim as u32,
            eps,
            _pad: 0,
        };

        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RMSNorm Uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RMSNorm BindGroup"),
            layout: &ctx.rmsnorm_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RMSNorm Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RMSNorm Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.rmsnorm_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(num_tokens as u32, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        Ok(out)
    }

    /// Native Fused GPU SwiGLU Activation
    pub fn swiglu(&self, ctx: &GpuContext, gate: &Self) -> Result<Self> {
        if self.numel != gate.numel {
            return Err(HarnessError::ShapeMismatch {
                expected: self.shape.clone(),
                found: gate.shape.clone(),
            });
        }

        let out = Self::empty(ctx, self.shape.clone())?;

        let params = ActUniform {
            num_elements: self.numel as u32,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };

        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("SwiGLU Uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("SwiGLU BindGroup"),
            layout: &ctx.swiglu_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: gate.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("SwiGLU Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("SwiGLU Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.swiglu_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let gx = (self.numel as u32).div_ceil(256);
            cpass.dispatch_workgroups(gx, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        Ok(out)
    }

    /// Native GPU Rotary Positional Embeddings (RoPE)
    pub fn rope(
        &self,
        ctx: &GpuContext,
        num_heads: usize,
        head_dim: usize,
        start_pos: usize,
        freq_base: f32,
    ) -> Result<Self> {
        let num_tokens = self.numel / (num_heads * head_dim);
        let out = Self::empty(ctx, self.shape.clone())?;

        let params = RoPEUniform {
            num_tokens: num_tokens as u32,
            num_heads: num_heads as u32,
            head_dim: head_dim as u32,
            start_pos: start_pos as u32,
            freq_base,
            _pad0: 0.0,
            _pad1: 0.0,
            _pad2: 0.0,
        };

        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RoPE Uniform"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("RoPE BindGroup"),
            layout: &ctx.rope_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RoPE Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RoPE Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.rope_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            let total_pairs = (num_tokens * num_heads * (head_dim / 2)) as u32;
            let gx = total_pairs.div_ceil(256);
            cpass.dispatch_workgroups(gx, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        Ok(out)
    }

    /// Native High-Speed GPU Vector-Matrix Multiplication for M=1 token decoding: y = x * W
    /// x: [1, K] or [K], W: [K, N] -> y: [1, N]
    pub fn gemv(&self, ctx: &GpuContext, weight: &Self) -> Result<Self> {
        let k = self.numel;
        if weight.shape.len() != 2 {
            return Err(HarnessError::InvalidShape(format!(
                "GPU GEMV weight must be 2D [K, N], got {:?}",
                weight.shape
            )));
        }

        let k2 = weight.shape[0];
        let n = weight.shape[1];

        if k != k2 {
            return Err(HarnessError::ShapeMismatch {
                expected: vec![1, k],
                found: vec![k2, n],
            });
        }

        let out = Self::empty(ctx, vec![1, n])?;

        let dims = GemvUniform {
            k: k as u32,
            n: n as u32,
            _pad0: 0,
            _pad1: 0,
        };

        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GEMV Dims Uniform"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GEMV BindGroup"),
            layout: &ctx.gemv_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GEMV Command Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GEMV Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.gemv_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(n as u32, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        Ok(out)
    }

    /// Native Ultra-Fast 4-Bit Quantized GEMV for M=1 token decoding: y = x * W_q4
    /// Directly dequantizes 4-bit weights in-register on GPU, eliminating 81.25% memory bus bandwidth
    pub fn gemv_q4(&self, ctx: &GpuContext, weight: &GpuQ4Tensor) -> Result<Self> {
        let k = self.numel;
        if k != weight.k {
            return Err(HarnessError::ShapeMismatch {
                expected: vec![1, k],
                found: vec![weight.k, weight.n],
            });
        }

        let n = weight.n;
        let blocks_per_col = k.div_ceil(32);
        let out = Self::empty(ctx, vec![1, n])?;

        let dims = Q4GemvUniform {
            k: k as u32,
            n: n as u32,
            blocks_per_col: blocks_per_col as u32,
            _pad: 0,
        };

        let uniform_buf = ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("GEMV Q4 Dims Uniform"),
            contents: bytemuck::bytes_of(&dims),
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let bind_group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("GEMV Q4 BindGroup"),
            layout: &ctx.gemv_q4_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: weight.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: out.buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buf.as_entire_binding(),
                },
            ],
        });

        let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("GEMV Q4 Command Encoder"),
        });

        {
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("GEMV Q4 Compute Pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&ctx.gemv_q4_pipeline);
            cpass.set_bind_group(0, &bind_group, &[]);
            cpass.dispatch_workgroups(n as u32, 1, 1);
        }

        ctx.queue.submit(Some(encoder.finish()));
        Ok(out)
    }
}

