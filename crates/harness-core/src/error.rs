use thiserror::Error;

pub type Result<T> = std::result::Result<T, HarnessError>;

#[derive(Error, Debug)]
pub enum HarnessError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("Device error: {0}")]
    Device(String),

    #[error("Out of memory on {device}: requested {requested_bytes} bytes, available {available_bytes} bytes")]
    OutOfMemory {
        device: String,
        requested_bytes: usize,
        available_bytes: usize,
    },

    #[error("Tensor shape mismatch: expected {expected:?}, found {found:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        found: Vec<usize>,
    },

    #[error("Invalid tensor shape or buffer bounds: {0}")]
    InvalidShape(String),

    #[error("Unsupported DType conversion from {from:?} to {to:?}")]
    UnsupportedDTypeConversion {
        from: crate::dtype::DType,
        to: crate::dtype::DType,
    },

    #[error("Model load failure: {0}")]
    ModelLoad(String),

    #[error("Quantization error: {0}")]
    Quantization(String),

    #[error("Attention error: {0}")]
    Attention(String),

    #[error("Pipeline scheduling error: {0}")]
    Pipeline(String),

    #[error("Constrained decoding violation: {0}")]
    ConstrainedDecoding(String),

    #[error("Anti-hallucination guardrail triggered: {0}")]
    SafetyGuardrail(String),

    #[error("Inference internal error: {0}")]
    Internal(String),
}
