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
            504.0  // GDDR6 on RTX 3070/4060
        } else {
            64.0   // Dual-channel host DDR5
        };

        // 4. Formulate optimal execution strategy for 70B models
        let (recommended_strategy, max_context) = if is_unified_memory {
            // On Apple Silicon, RAM IS VRAM! Zero-copy GPU access without PCIe bus transfers
            if host_ram_gb >= 64.0 {
                ("Apple Silicon UMA: 70B 100% Resident in Unified RAM (Zero-PCIe Overhead, 800GB/s)".into(), 131072)
            } else if host_ram_gb >= 36.0 {
                ("Apple Silicon UMA: 70B Q4_K_M Zero-Copy Unified RAM + FP8 KV Cache".into(), 65536)
            } else if host_ram_gb >= 16.0 {
                ("Apple Silicon UMA: High-Bandwidth Layer Streaming (Zero-PCIe Bottleneck)".into(), 32768)
            } else {
                ("Apple Silicon UMA: Compact NF4 Dynamic Unified Paged Cache".into(), 16384)
            }
        } else if vram_gb >= 40.0 {
            ("Full VRAM Resident (Dense/MoE Zero-Offload)".into(), 131072)
        } else if vram_gb >= 24.0 {
            ("Hybrid Offload: 70B Q4 Resident + FP8 KV Cache".into(), 65536)
        } else if vram_gb >= 8.0 || host_ram_gb >= 32.0 {
            ("Temporal Layer Streaming (70B on 8GB VRAM Ping-Pong DMA)".into(), 32768)
        } else {
            ("Aggressive Quantization (NF4) + CPU Paged Swap".into(), 8192)
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
}

fn detect_system_ram_gb() -> f32 {
    #[cfg(target_os = "windows")]
    {
        #[repr(C)]
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
