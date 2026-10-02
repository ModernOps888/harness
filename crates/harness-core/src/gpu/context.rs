use crate::gpu::kernels::*;
use crate::{HarnessError, Result};
use std::sync::Arc;

/// Native GPU Execution Context managing hardware device, queues, and compute pipelines
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub adapter_name: String,
    pub backend: String,
    pub max_buffer_size: u64,

    // Compiled Compute Pipelines
    pub gemm_pipeline: wgpu::ComputePipeline,
    pub gemm_layout: wgpu::BindGroupLayout,

    pub gemv_pipeline: wgpu::ComputePipeline,
    pub gemv_layout: wgpu::BindGroupLayout,

    pub gemv_q4_pipeline: wgpu::ComputePipeline,
    pub gemv_q4_layout: wgpu::BindGroupLayout,

    pub rmsnorm_pipeline: wgpu::ComputePipeline,
    pub rmsnorm_layout: wgpu::BindGroupLayout,

    pub swiglu_pipeline: wgpu::ComputePipeline,
    pub swiglu_layout: wgpu::BindGroupLayout,

    pub rope_pipeline: wgpu::ComputePipeline,
    pub rope_layout: wgpu::BindGroupLayout,
}

impl GpuContext {
    /// Initialize high-performance native GPU context
    pub fn new() -> Result<Self> {
        pollster::block_on(Self::new_async())
    }

    /// Async initializer requesting discrete GPU adapter
    pub async fn new_async() -> Result<Self> {
        let instance = wgpu::Instance::default();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| HarnessError::Device("No compatible GPU adapter found".into()))?;

        let info = adapter.get_info();
        let adapter_name = info.name.clone();
        let backend = format!("{:?}", info.backend);

        let adapter_limits = adapter.limits();
        let requested_limits = wgpu::Limits {
            max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
            max_buffer_size: adapter_limits.max_buffer_size,
            max_compute_workgroup_size_x: 256,
            max_compute_workgroup_size_y: 256,
            max_compute_workgroup_size_z: 64,
            max_compute_invocations_per_workgroup: 256,
            ..wgpu::Limits::default()
        };

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("HARNESS GPU Compute Engine"),
                    required_features: wgpu::Features::empty(),
                    required_limits: requested_limits,
                    memory_hints: wgpu::MemoryHints::Performance,
                },
                None,
            )
            .await
            .map_err(|e| HarnessError::Device(format!("Failed to request GPU device: {}", e)))?;

        let device = Arc::new(device);
        let queue = Arc::new(queue);
        let max_buffer_size = adapter_limits.max_buffer_size;

        // 1. GEMM Pipeline
        let (gemm_pipeline, gemm_layout) = Self::create_gemm_pipeline(&device)?;

        // 1b. GEMV Decoding Pipeline
        let (gemv_pipeline, gemv_layout) = Self::create_gemv_pipeline(&device)?;

        // 1c. GEMV Q4 Quantized Pipeline
        let (gemv_q4_pipeline, gemv_q4_layout) = Self::create_gemv_q4_pipeline(&device)?;

        // 2. RMSNorm Pipeline
        let (rmsnorm_pipeline, rmsnorm_layout) = Self::create_rmsnorm_pipeline(&device)?;

        // 3. SwiGLU Pipeline
        let (swiglu_pipeline, swiglu_layout) = Self::create_swiglu_pipeline(&device)?;

        // 4. RoPE Pipeline
        let (rope_pipeline, rope_layout) = Self::create_rope_pipeline(&device)?;

        Ok(Self {
            device,
            queue,
            adapter_name,
            backend,
            max_buffer_size,
            gemm_pipeline,
            gemm_layout,
            gemv_pipeline,
            gemv_layout,
            gemv_q4_pipeline,
            gemv_q4_layout,
            rmsnorm_pipeline,
            rmsnorm_layout,
            swiglu_pipeline,
            swiglu_layout,
            rope_pipeline,
            rope_layout,
        })
    }

    fn create_gemm_pipeline(
        device: &wgpu::Device,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GEMM Shader Module"),
            source: wgpu::ShaderSource::Wgsl(GEMM_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GEMM Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GEMM Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GEMM Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gemm_tiled"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_gemv_pipeline(
        device: &wgpu::Device,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GEMV Shader Module"),
            source: wgpu::ShaderSource::Wgsl(GEMV_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GEMV Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GEMV Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GEMV Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gemv_decode"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_gemv_q4_pipeline(
        device: &wgpu::Device,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GEMV Q4 Shader Module"),
            source: wgpu::ShaderSource::Wgsl(GEMV_Q4_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GEMV Q4 Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("GEMV Q4 Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GEMV Q4 Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("gemv_q4_decode"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_rmsnorm_pipeline(
        device: &wgpu::Device,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RMSNorm Shader Module"),
            source: wgpu::ShaderSource::Wgsl(RMSNORM_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RMSNorm Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RMSNorm Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RMSNorm Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("rms_norm"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_swiglu_pipeline(
        device: &wgpu::Device,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("SwiGLU Shader Module"),
            source: wgpu::ShaderSource::Wgsl(SWIGLU_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("SwiGLU Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("SwiGLU Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("SwiGLU Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("swiglu_forward"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }

    fn create_rope_pipeline(
        device: &wgpu::Device,
    ) -> Result<(wgpu::ComputePipeline, wgpu::BindGroupLayout)> {
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RoPE Shader Module"),
            source: wgpu::ShaderSource::Wgsl(ROPE_SHADER.into()),
        });

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RoPE Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("RoPE Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RoPE Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("rope_forward"),
            compilation_options: Default::default(),
            cache: None,
        });

        Ok((pipeline, layout))
    }
}
