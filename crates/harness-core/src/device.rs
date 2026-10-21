use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Available compute backend devices
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum Device {
    #[default]
    Cpu,
    Cuda(usize),  // NVIDIA CUDA index
    Rocm(usize),  // AMD ROCm/HIP index
    Metal(usize), // Apple Silicon Metal index
    Vulkan(usize),// Cross-platform Vulkan compute index
}

impl std::fmt::Display for Device {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Cpu => write!(f, "CPU (SIMD)"),
            Self::Cuda(idx) => write!(f, "NVIDIA CUDA:{}", idx),
            Self::Rocm(idx) => write!(f, "AMD ROCm:{}", idx),
            Self::Metal(idx) => write!(f, "Apple Metal:{}", idx),
            Self::Vulkan(idx) => write!(f, "Vulkan:{}", idx),
        }
    }
}

/// Detailed multi-platform hardware auto-detection profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareProfile {
    pub os: String,
    pub arch: String,
    pub host_ram_gb: f32,
    pub accelerator_name: String,
    pub accelerator_device: Device,
    pub vram_gb: f32,
    pub is_unified_memory: bool,
    pub memory_bandwidth_gbps: f32,
    pub recommended_70b_strategy: String,
    pub max_supported_context: usize,
}

impl HardwareProfile {
    /// Detect host operating system, memory topology, and hardware accelerators
    pub fn auto_detect() -> Self {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();

        // 1. Detect Host RAM
        let host_ram_gb = detect_system_ram_gb();

        // 2. Detect Accelerators (NVIDIA, AMD, Apple Metal)
        let (accelerator_name, accelerator_device, vram_gb) = detect_accelerator(&os);

        // 3. Apple Silicon Unified Memory Architecture (UMA) Detection (M3, M4, M5 generations)
        let is_unified_memory = os == "macos";
        let memory_bandwidth_gbps = if is_unified_memory {
            // M3/M4/M5 UMA Bandwidth Matrix:
            // Base (M3: 100, M4: 120, M5: 153 GB/s)
            // Pro (M3 Pro: 150, M4 Pro: 273, M5 Pro: 350 GB/s)
            // Max (M3 Max: 400, M4 Max: 546, M5 Max: 650 GB/s)
            // Ultra (M2/M4/M5 Ultra: 800 - 1300+ GB/s)
            if host_ram_gb >= 128.0 { 1092.0 } // Ultra tier
            else if host_ram_gb >= 64.0 { 546.0 } // Max tier (M4 Max 40-core GPU)
            else if host_ram_gb >= 36.0 { 400.0 } // High Pro / Max baseline
            else if host_ram_gb >= 24.0 { 273.0 } // M4 Pro (273 GB/s via 8533 MT/s LPDDR5X)
            else if host_ram_gb >= 16.0 { 120.0 } // M4 base (120 GB/s) / M3 Pro
            else { 100.0 }                        // M3 base (100 GB/s)
        } else if vram_gb >= 24.0 {
            1008.0 // GDDR6X on RTX 3090/4090
        } else if vram_gb >= 8.0 {
            if accelerator_name.contains("5060") {
                448.0 // GDDR7 on RTX 5060 (128-bit @ 28 Gbps = 448 GB/s)
            } else {
                504.0 // GDDR6 on RTX 3070/4060 Ti
            }
        } else {
            64.0   // Dual-channel host DDR5
        };

        // 4. Formulate optimal execution strategy for 27B-35B Coder & 70B models
        let (recommended_strategy, max_context) = if is_unified_memory {
            // On Apple Silicon, RAM IS VRAM! Zero-copy GPU access without PCIe bus transfers
            if host_ram_gb >= 64.0 {
                ("Apple Silicon UMA: 32B Coder & 70B 100% Resident in Unified RAM (Zero-PCIe Overhead, 800GB/s)".into(), 131072)
            } else if host_ram_gb >= 36.0 {
                ("Apple Silicon UMA: 32B Coder 100% Resident + 1.5B Speculative Drafter (Zero-Copy Metal)".into(), 65536)
            } else if host_ram_gb >= 24.0 {
                ("Apple Silicon UMA: 27B-32B Coder Q4_K_M Resident + Q8_0/Q8_0 KV Cache".into(), 32768)
            } else {
                ("Apple Silicon UMA: 14B Coder Q4_K_S / 30B-A3B MoE Resident + Compact KV Cache".into(), 16384)
            }
        } else if vram_gb >= 24.0 {
            ("24GB+ VRAM: 32B Coder 100% GPU Resident (-ngl 64) + 1.5B Drafter + Q8_0 KV Cache".into(), 65536)
        } else if vram_gb >= 16.0 {
            ("16GB VRAM Hybrid: 32B Coder (-ngl 44) + 100% VRAM 1.5B Drafter (-ngld 99) + Q8_0 KV".into(), 32768)
        } else if vram_gb >= 8.0 || host_ram_gb >= 32.0 {
            ("Qwen3-Coder-30B-A3B 32k Agent Mode (-ngl 99 --n-cpu-moe 37 -c 32768 -ctk q8_0 -ctv q8_0 -fa on -b 2048 -ub 1024 --jinja)".into(), 32768)
        } else {
            ("Compact Coder Tier: 14B Q4_K_S / 30B-A3B Sparse MoE + Q8_0 KV Cache".into(), 8192)
        };

        Self {
            os,
            arch,
            host_ram_gb,
            accelerator_name,
            accelerator_device,
            vram_gb,
            is_unified_memory,
            memory_bandwidth_gbps,
            recommended_70b_strategy: recommended_strategy,
            max_supported_context: max_context,
        }
    }

    /// Compute an exact hardware-grounded GPU/CPU offload & speculative decoding plan for 27B-35B Coder models
    pub fn plan_coder_offload(
        &self,
        config: &crate::config::ModelConfig,
        draft_vram_mb: usize,
        ctx_len: usize,
    ) -> CoderOffloadPlan {
        let total_layers = config.num_hidden_layers;
        let per_layer_bytes = config.per_layer_weight_bytes_q4();
        let per_layer_weight_mb = per_layer_bytes as f32 / (1024.0 * 1024.0);

        let kv_q8_bytes = config.kv_cache_bytes_per_token(8, 8) * ctx_len;
        let kv_q4_bytes = config.kv_cache_bytes_per_token(8, 4) * ctx_len;
        let kv_cache_q8_mb = kv_q8_bytes.div_ceil(1024 * 1024);
        let kv_cache_q4_mb = kv_q4_bytes.div_ceil(1024 * 1024);

        let vram_total_mb = (self.vram_gb * 1024.0) as usize;
        // Reserve ~480 MB for CUDA/Vulkan context, FlashAttention workspace, and output lm_head staging
        let compute_scratch_mb = 480usize;
        let reserved_mb = compute_scratch_mb + draft_vram_mb + kv_cache_q4_mb;
        let usable_for_target_mb = vram_total_mb.saturating_sub(reserved_mb);

        let raw_ngl = (usable_for_target_mb as f32 / per_layer_weight_mb.max(1.0)).floor() as usize;
        let gpu_layers_ngl = if self.is_unified_memory && self.host_ram_gb >= 24.0 {
            total_layers
        } else {
            raw_ngl.min(total_layers)
        };
        let cpu_layers = total_layers.saturating_sub(gpu_layers_ngl);
        let gpu_weights_mb = (gpu_layers_ngl as f32 * per_layer_weight_mb).round() as usize;
        let ddr4_traffic_gb_per_pass = (cpu_layers as f32 * per_layer_weight_mb) / 1024.0;

        let draft_gpu_layers_ngld = if draft_vram_mb > 0 { 99 } else { 0 };
        let spec_draft_max = if draft_vram_mb > 0 { 8 } else { 0 };
        let recommended_threads = std::thread::available_parallelism()
            .map(|n| (n.get() / 2).max(4))
            .unwrap_or(6);

        let llama_server_flags = format!(
            "-ngl {} -ngld {} --spec-draft-n-max {} -c {} -t {} -ctk q8_0 -ctv q4_0 -fa on -b 512 -ub 256",
            gpu_layers_ngl, draft_gpu_layers_ngld, spec_draft_max, ctx_len, recommended_threads
        );

        CoderOffloadPlan {
            total_layers,
            gpu_layers_ngl,
            cpu_layers,
            draft_gpu_layers_ngld,
            draft_vram_mb,
            kv_cache_q8_mb,
            kv_cache_q4_mb,
            per_layer_weight_mb,
            gpu_weights_mb,
            ddr4_traffic_gb_per_pass,
            spec_draft_max,
            recommended_threads,
            llama_server_flags,
        }
    }
}

/// Hardware-grounded offload & speculative decoding execution plan for 27B-35B Coder models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoderOffloadPlan {
    pub total_layers: usize,
    pub gpu_layers_ngl: usize,
    pub cpu_layers: usize,
    pub draft_gpu_layers_ngld: usize,
    pub draft_vram_mb: usize,
    pub kv_cache_q8_mb: usize,
    pub kv_cache_q4_mb: usize,
    pub per_layer_weight_mb: f32,
    pub gpu_weights_mb: usize,
    pub ddr4_traffic_gb_per_pass: f32,
    pub spec_draft_max: usize,
    pub recommended_threads: usize,
    pub llama_server_flags: String,
}

fn detect_system_ram_gb() -> f32 {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        struct MEMORYSTATUSEX {
            dw_length: u32,
            dw_memory_load: u32,
            ull_total_phys: u64,
            ull_avail_phys: u64,
            ull_total_page_file: u64,
            ull_avail_page_file: u64,
            ull_total_virtual: u64,
            ull_avail_virtual: u64,
            ull_avail_extended_virtual: u64,
        }

        extern "system" {
            fn GlobalMemoryStatusEx(stat: *mut MEMORYSTATUSEX) -> i32;
        }

        let mut mem = MEMORYSTATUSEX {
            dw_length: std::mem::size_of::<MEMORYSTATUSEX>() as u32,
            dw_memory_load: 0,
            ull_total_phys: 0,
            ull_avail_phys: 0,
            ull_total_page_file: 0,
            ull_avail_page_file: 0,
            ull_total_virtual: 0,
            ull_avail_virtual: 0,
            ull_avail_extended_virtual: 0,
        };

        let ok = unsafe { GlobalMemoryStatusEx(&mut mem) };
        if ok != 0 && mem.ull_total_phys > 0 {
            return (mem.ull_total_phys as f64 / (1024.0 * 1024.0 * 1024.0)) as f32;
        }
        32.0
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<f64>() {
                            return (kb / (1024.0 * 1024.0)) as f32;
                        }
                    }
                }
            }
        }
        32.0
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(s) = std::str::from_utf8(&output.stdout) {
                if let Ok(bytes) = s.trim().parse::<f64>() {
                    return (bytes / (1024.0 * 1024.0 * 1024.0)) as f32;
                }
            }
        }
        32.0 // Apple Silicon Unified Memory fallback
    }
    #[cfg(not(any(target_os = "windows", target_os = "linux", target_os = "macos")))]
    {
        16.0
    }
}

fn detect_accelerator(os: &str) -> (String, Device, f32) {
    if os == "macos" {
        let ram = detect_system_ram_gb();
        return (format!("Apple Silicon Metal Unified Memory ({:.0}GB UMA)", ram), Device::Metal(0), ram);
    }

    // Try dynamic nvidia-smi probe first for exact name and VRAM
    if let Ok(output) = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,memory.total", "--format=csv,noheader,nounits"])
        .output()
    {
        if output.status.success() {
            if let Ok(s) = std::str::from_utf8(&output.stdout) {
                if let Some(line) = s.lines().next() {
                    let parts: Vec<&str> = line.split(',').map(|p| p.trim()).collect();
                    if parts.len() >= 2 {
                        let gpu_name = parts[0].to_string();
                        let vram_mb = parts[1].parse::<f32>().unwrap_or(8192.0);
                        let vram_gb = (vram_mb / 1024.0 * 10.0).round() / 10.0;
                        return (gpu_name, Device::Cuda(0), vram_gb);
                    }
                }
            }
        }
    }

    // Check for NVIDIA CUDA presence via environment or driver paths
    if std::env::var("CUDA_VISIBLE_DEVICES").is_ok()
        || std::path::Path::new("/usr/local/cuda").exists()
        || std::path::Path::new("C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA").exists()
        || std::path::Path::new("/dev/nvidia0").exists()
    {
        return ("NVIDIA RTX Architecture (CUDA Compute 8.9+)".into(), Device::Cuda(0), 8.0);
    }

    // Check for AMD ROCm presence
    if std::path::Path::new("/opt/rocm").exists() || std::path::Path::new("/dev/kfd").exists() {
        return ("AMD Radeon / Instinct (ROCm HIP)".into(), Device::Rocm(0), 16.0);
    }

    // High-performance CPU with AVX-512 / AVX2 SIMD fallback
    ("Host CPU (AVX-512 / AVX2 Vector Engine)".into(), Device::Cpu, 8.0)
}

/// Real-time memory allocation statistics for a device
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceMemoryStats {
    pub device: Device,
    pub total_bytes: usize,
    pub allocated_bytes: usize,
    pub peak_bytes: usize,
    pub kv_cache_bytes: usize,
    pub weights_bytes: usize,
    pub activations_bytes: usize,
}

impl DeviceMemoryStats {
    pub fn free_bytes(&self) -> usize {
        self.total_bytes.saturating_sub(self.allocated_bytes)
    }

    pub fn usage_percentage(&self) -> f32 {
        if self.total_bytes == 0 {
            0.0
        } else {
            (self.allocated_bytes as f32 / self.total_bytes as f32) * 100.0
        }
    }
}

/// Global device manager tracking allocations across host and device memories
#[derive(Debug, Clone)]
pub struct DeviceManager {
    device: Device,
    allocated_bytes: Arc<AtomicUsize>,
    peak_bytes: Arc<AtomicUsize>,
    total_capacity: usize,
    kv_allocated: Arc<AtomicUsize>,
    weight_allocated: Arc<AtomicUsize>,
}

impl DeviceManager {
    pub fn new(device: Device, total_capacity: usize) -> Self {
        Self {
            device,
            allocated_bytes: Arc::new(AtomicUsize::new(0)),
            peak_bytes: Arc::new(AtomicUsize::new(0)),
            total_capacity,
            kv_allocated: Arc::new(AtomicUsize::new(0)),
            weight_allocated: Arc::new(AtomicUsize::new(0)),
        }
    }

    pub fn detect_primary() -> Self {
        let profile = HardwareProfile::auto_detect();
        let total_vram = (profile.vram_gb * 1024.0 * 1024.0 * 1024.0) as usize;
        Self::new(profile.accelerator_device, total_vram)
    }

    pub fn allocate(&self, bytes: usize, is_kv: bool, is_weight: bool) -> crate::error::Result<()> {
        let current = self.allocated_bytes.load(Ordering::Relaxed);
        if current + bytes > self.total_capacity && self.total_capacity > 0 {
            return Err(crate::error::HarnessError::OutOfMemory {
                device: self.device.to_string(),
                requested_bytes: bytes,
                available_bytes: self.total_capacity.saturating_sub(current),
            });
        }

        let new_total = self.allocated_bytes.fetch_add(bytes, Ordering::SeqCst) + bytes;
        let mut peak = self.peak_bytes.load(Ordering::Relaxed);
        while new_total > peak {
            match self.peak_bytes.compare_exchange_weak(
                peak,
                new_total,
                Ordering::SeqCst,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => peak = actual,
            }
        }

        if is_kv {
            self.kv_allocated.fetch_add(bytes, Ordering::Relaxed);
        } else if is_weight {
            self.weight_allocated.fetch_add(bytes, Ordering::Relaxed);
        }

        Ok(())
    }

    pub fn deallocate(&self, bytes: usize, is_kv: bool, is_weight: bool) {
        self.allocated_bytes.fetch_sub(bytes, Ordering::SeqCst);
        if is_kv {
            self.kv_allocated.fetch_sub(bytes, Ordering::Relaxed);
        } else if is_weight {
            self.weight_allocated.fetch_sub(bytes, Ordering::Relaxed);
        }
    }

    pub fn snapshot(&self) -> DeviceMemoryStats {
        let allocated = self.allocated_bytes.load(Ordering::Relaxed);
        let peak = self.peak_bytes.load(Ordering::Relaxed);
        let kv = self.kv_allocated.load(Ordering::Relaxed);
        let weights = self.weight_allocated.load(Ordering::Relaxed);
        let activations = allocated.saturating_sub(kv + weights);

        DeviceMemoryStats {
            device: self.device.clone(),
            total_bytes: self.total_capacity,
            allocated_bytes: allocated,
            peak_bytes: peak,
            kv_cache_bytes: kv,
            weights_bytes: weights,
            activations_bytes: activations,
        }
    }
}
