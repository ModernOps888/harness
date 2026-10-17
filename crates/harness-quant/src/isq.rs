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

    /// Analyze tensor outlier crest factor (max_abs / rms), kurtosis, and layer topology
    /// to assign optimal quantization precision (mirroring Q4_K_M / IQ4_XS mixed-precision allocation).
    pub fn profile_layer(&self, layer_name: &str, weights: &Tensor) -> Result<LayerQuantProfile> {
        let f32_data = weights.as_f32_slice()?;
        let n = f32_data.len().max(1) as f32;
        let mean = f32_data.iter().sum::<f32>() / n;
        let mut sum_sq = 0.0f32;
        let mut variance_acc = 0.0f32;
        let mut max_abs = 0.0f32;

        for &x in f32_data {
            let abs_x = x.abs();
            if abs_x > max_abs {
                max_abs = abs_x;
            }
            sum_sq += x * x;
            let diff = x - mean;
            variance_acc += diff * diff;
        }

        let rms = (sum_sq / n).sqrt();
        let variance = variance_acc / n;
        let crest_factor = if rms > 1e-8 { max_abs / rms } else { 1.0 };

        // Embedding, output/lm_head, norms, and Q4_K_M sensitive projections (attn_v, ffn_down)
        let is_critical_layer = layer_name.contains("embed")
            || layer_name.contains("lm_head")
            || layer_name.contains("output")
            || layer_name.contains("norm")
            || layer_name.contains("layers.0.")
            || layer_name.contains(".final_");

        let is_high_salience_proj = layer_name.contains("attn_v")
            || layer_name.contains("v_proj")
            || layer_name.contains("ffn_down")
            || layer_name.contains("down_proj");

        let sensitivity = if is_critical_layer {
            0.95
        } else if is_high_salience_proj || crest_factor > 6.0 || variance > 0.05 {
            0.78
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

