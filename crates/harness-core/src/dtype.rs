use serde::{Deserialize, Serialize};

/// Precision and quantization data types supported by HARNESS
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    /// 32-bit standard IEEE-754 float
    F32,
    /// 16-bit half precision float
    F16,
    /// 16-bit brain floating point
    BF16,
    /// 8-bit FP8 (E4M3) for maximum forward pass accuracy on modern hardware
    FP8E4M3,
    /// 8-bit FP8 (E5M2) with higher dynamic range
    FP8E5M2,
    /// 8-bit signed integer
    I8,
    /// 4-bit integer packed format
    I4,
    /// 4-bit NormalFloat (NF4) for information-theoretically optimal weight quantization
    NF4,
    /// 4-bit floating point (FP4) for ultra-low KV-cache and weights
    FP4,
}

impl DType {
    /// Size in bits per scalar element
    pub const fn bit_size(&self) -> usize {
        match self {
            Self::F32 => 32,
            Self::F16 | Self::BF16 => 16,
            Self::FP8E4M3 | Self::FP8E5M2 | Self::I8 => 8,
            Self::I4 | Self::NF4 | Self::FP4 => 4,
        }
    }

    /// Size in bytes for N elements (rounded up for sub-byte types)
    pub const fn byte_size_for_elements(&self, count: usize) -> usize {
        let total_bits = count * self.bit_size();
        total_bits.div_ceil(8)
    }

    /// Returns whether this datatype is a quantized low-precision type (<= 8 bits)
    pub const fn is_quantized(&self) -> bool {
        self.bit_size() <= 8 && !matches!(self, Self::F32)
    }

    /// Human readable name
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::F32 => "float32",
            Self::F16 => "float16",
            Self::BF16 => "bfloat16",
            Self::FP8E4M3 => "fp8-e4m3",
            Self::FP8E5M2 => "fp8-e5m2",
            Self::I8 => "int8",
            Self::I4 => "int4",
            Self::NF4 => "nf4",
            Self::FP4 => "fp4",
        }
    }
}
