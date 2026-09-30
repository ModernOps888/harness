use harness_core::{DType, Result, Tensor};

#[derive(Debug, Clone, Copy)]
pub enum QuantTarget {
    MaxSpeed,      // Aggressive 4-bit on all possible layers
    Balanced,      // Sensitive layers in FP8, others in Q4
    MaxQuality,    // FP8 throughout
}

#[derive(Debug, Clone)]
pub struct LayerQuantProfile {
    pub layer_name: String,
    pub sensitivity: f32,
    pub target_dtype: DType,
}

pub struct ISQEngine {
    target: QuantTarget,
}

impl ISQEngine {
    pub fn new(target: QuantTarget) -> Self {
        Self { target }
    }

    /// Analyze tensor variance and gradient sensitivity to assign optimal quantization precision
    pub fn profile_layer(&self, layer_name: &str, weights: &Tensor) -> Result<LayerQuantProfile> {
        let f32_data = weights.as_f32_slice()?;
        let mean = f32_data.iter().sum::<f32>() / f32_data.len() as f32;
        let variance = f32_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / f32_data.len() as f32;

        // Embedding layers and lm_head have higher outlier sensitivity
        let is_critical_layer = layer_name.contains("embed")
            || layer_name.contains("lm_head")
            || layer_name.contains("norm")
            || layer_name.contains("layers.0.")
            || layer_name.contains(".final_");

        let sensitivity = if is_critical_layer {
            0.95
        } else if variance > 0.05 {
            0.75
        } else {
            0.35
        };

        let target_dtype = match (self.target, sensitivity > 0.7) {
            (QuantTarget::MaxQuality, _) => DType::FP8E4M3,
            (QuantTarget::Balanced, true) => DType::FP8E4M3,
            (QuantTarget::Balanced, false) => DType::I4,
            (QuantTarget::MaxSpeed, true) => DType::I8,
            (QuantTarget::MaxSpeed, false) => DType::I4,
        };

        Ok(LayerQuantProfile {
            layer_name: layer_name.to_string(),
            sensitivity,
            target_dtype,
        })
    }
}
