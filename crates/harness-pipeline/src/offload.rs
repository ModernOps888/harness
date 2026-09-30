use harness_core::{Device, DeviceManager, ModelConfig, Result};
use std::ops::Range;
use tracing::info;

#[derive(Debug, Clone)]
pub struct OffloadProfile {
    pub total_layers: usize,
    pub gpu_layers: Range<usize>,
    pub cpu_layers: Range<usize>,
    pub estimated_vram_usage_bytes: usize,
    pub estimated_ram_usage_bytes: usize,
}

pub struct HybridOffloader {
    pub profile: OffloadProfile,
    pub device_mgr: DeviceManager,
}

impl HybridOffloader {
    pub fn plan(config: &ModelConfig, available_vram_bytes: usize) -> OffloadProfile {
        let total_layers = config.num_hidden_layers;
        // Estimate bytes per layer (weights + KV cache overhead)
        let bytes_per_layer = (config.hidden_size * config.intermediate_size * 3 * 2) / 4; // ~Q4 size

        let max_gpu_fit = (available_vram_bytes * 8 / 10) / bytes_per_layer.max(1);
        let num_gpu = max_gpu_fit.min(total_layers);

        let gpu_layers = 0..num_gpu;
        let cpu_layers = num_gpu..total_layers;

        info!(
            total = total_layers,
            gpu_layers = ?gpu_layers,
            cpu_layers = ?cpu_layers,
            "Calculated Hybrid GPU<->CPU layer distribution"
        );

        OffloadProfile {
            total_layers,
            gpu_layers,
            cpu_layers: cpu_layers.clone(),
            estimated_vram_usage_bytes: num_gpu * bytes_per_layer,
            estimated_ram_usage_bytes: (total_layers - num_gpu) * bytes_per_layer,
        }
    }
}
