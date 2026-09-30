//! # HARNESS Core
//!
//! Foundational abstractions for the HARNESS inference engine:
//! - Device & memory management (VRAM / System RAM tracking)
//! - Precision datatypes (FP32 down to FP4/NF4)
//! - Memory-pooled Strided Tensors
//! - Universal LLM architecture configuration

pub mod config;
pub mod device;
pub mod dtype;
pub mod error;
pub mod tensor;

pub use config::{MoEConfig, ModelArchitecture, ModelConfig, QuantizationMode};
pub use device::{Device, DeviceManager, DeviceMemoryStats, HardwareProfile};
pub use dtype::DType;
pub use error::{HarnessError, Result};
pub use tensor::{Shape, Strides, Tensor};
