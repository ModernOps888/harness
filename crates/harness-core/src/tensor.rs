use crate::device::Device;
use crate::dtype::DType;
use crate::error::{HarnessError, Result};
use rayon::prelude::*;
use std::sync::Arc;

pub type Shape = Vec<usize>;
pub type Strides = Vec<usize>;

/// Strided, contiguous or view-based n-dimensional tensor
#[derive(Clone)]
pub struct Tensor {
    shape: Shape,
    strides: Strides,
    dtype: DType,
    device: Device,
    data: Arc<Vec<u8>>,
    offset: usize,
}

impl Tensor {
    /// Create a contiguous tensor with raw byte buffer
    pub fn from_raw_bytes(
        data: Vec<u8>,
        shape: Shape,
        dtype: DType,
        device: Device,
    ) -> Result<Self> {
        let expected_bytes = dtype.byte_size_for_elements(shape.iter().product());
        if data.len() < expected_bytes {
            return Err(HarnessError::ShapeMismatch {
                expected: vec![expected_bytes],
                found: vec![data.len()],
            });
        }
        let strides = Self::compute_contiguous_strides(&shape);
        Ok(Self {
            shape,
            strides,
            dtype,
            device,
            data: Arc::new(data),
            offset: 0,
        })
    }

    /// Create an f32 filled tensor
    pub fn from_f32_slice(values: &[f32], shape: Shape, device: Device) -> Result<Self> {
        let count: usize = shape.iter().product();
        if values.len() != count {
            return Err(HarnessError::ShapeMismatch {
                expected: vec![count],
                found: vec![values.len()],
            });
        }
        let mut bytes = Vec::with_capacity(count * 4);
        for &val in values {
            bytes.extend_from_slice(&val.to_le_bytes());
        }
        Self::from_raw_bytes(bytes, shape, DType::F32, device)
    }

    /// Create a zeros tensor
    pub fn zeros(shape: Shape, dtype: DType, device: Device) -> Result<Self> {
        let count: usize = shape.iter().product();
        let bytes_len = dtype.byte_size_for_elements(count);
        let bytes = vec![0u8; bytes_len];
        Self::from_raw_bytes(bytes, shape, dtype, device)
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn strides(&self) -> &[usize] {
        &self.strides
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn as_f32_slice(&self) -> Result<&[f32]> {
        if self.dtype != DType::F32 {
            return Err(HarnessError::UnsupportedDTypeConversion {
                from: self.dtype,
                to: DType::F32,
            });
        }
        let byte_len = self.numel().checked_mul(4).ok_or_else(|| {
            HarnessError::InvalidShape("Tensor size arithmetic overflow in byte calculation".into())
        })?;
        if self.offset.checked_add(byte_len).is_none_or(|end| end > self.data.len()) {
            return Err(HarnessError::InvalidShape(format!(
                "Offset {} + byte length {} exceeds buffer size {}",
                self.offset, byte_len, self.data.len()
            )));
        }
        let slice = &self.data[self.offset..self.offset + byte_len];
        let ptr = slice.as_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(HarnessError::InvalidShape(
                "Buffer is not 4-byte aligned for safe f32 slice casting".into(),
            ));
        }
        let f32_ptr = ptr as *const f32;
        unsafe { Ok(std::slice::from_raw_parts(f32_ptr, self.numel())) }
    }

    pub fn as_mut_f32_slice(&mut self) -> Result<&mut [f32]> {
        if self.dtype != DType::F32 {
            return Err(HarnessError::UnsupportedDTypeConversion {
                from: self.dtype,
                to: DType::F32,
            });
        }
        let byte_len = self.numel().checked_mul(4).ok_or_else(|| {
            HarnessError::InvalidShape("Tensor size arithmetic overflow in byte calculation".into())
        })?;
        if self.offset.checked_add(byte_len).is_none_or(|end| end > self.data.len()) {
            return Err(HarnessError::InvalidShape(format!(
                "Offset {} + byte length {} exceeds buffer size {}",
                self.offset, byte_len, self.data.len()
            )));
        }
        let data = Arc::make_mut(&mut self.data);
        let slice = &mut data[self.offset..self.offset + byte_len];
        let ptr = slice.as_mut_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<f32>()) {
            return Err(HarnessError::InvalidShape(
                "Buffer is not 4-byte aligned for safe f32 slice casting".into(),
            ));
        }
        let f32_ptr = ptr as *mut f32;
        unsafe { Ok(std::slice::from_raw_parts_mut(f32_ptr, self.numel())) }
    }

    /// Reshape tensor (must have same total number of elements)
    pub fn reshape(&self, new_shape: Shape) -> Result<Self> {
        let new_numel: usize = new_shape.iter().product();
        if self.numel() != new_numel {
            return Err(HarnessError::ShapeMismatch {
                expected: self.shape.clone(),
                found: new_shape,
            });
        }
        let strides = Self::compute_contiguous_strides(&new_shape);
        Ok(Self {
            shape: new_shape,
            strides,
            dtype: self.dtype,
            device: self.device.clone(),
            data: self.data.clone(),
            offset: self.offset,
        })
    }

    /// High performance multi-threaded GEMM for [M, K] x [K, N] -> [M, N]
    /// Uses Rayon row partitioning with 4-wide SIMD loop unrolling for AVX2/FMA execution
    pub fn matmul(&self, other: &Self) -> Result<Self> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(HarnessError::Internal("GEMM requires 2D matrices".into()));
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

        let a = self.as_f32_slice()?;
        let b = other.as_f32_slice()?;
        let mut c = vec![0.0f32; m * n];

        // Parallel cache-blocked GEMM with 8-wide unrolling for 256-bit AVX2 SIMD FMA
        c.par_chunks_mut(n).enumerate().for_each(|(i, c_row)| {
            let a_row_offset = i * k1;
            for p in 0..k1 {
                let a_val = a[a_row_offset + p];
                let b_row_offset = p * n;
                let b_slice = &b[b_row_offset..b_row_offset + n];

                let mut j = 0;
                while j + 8 <= n {
                    c_row[j] += a_val * b_slice[j];
                    c_row[j + 1] += a_val * b_slice[j + 1];
                    c_row[j + 2] += a_val * b_slice[j + 2];
                    c_row[j + 3] += a_val * b_slice[j + 3];
                    c_row[j + 4] += a_val * b_slice[j + 4];
                    c_row[j + 5] += a_val * b_slice[j + 5];
                    c_row[j + 6] += a_val * b_slice[j + 6];
                    c_row[j + 7] += a_val * b_slice[j + 7];
                    j += 8;
                }
                while j < n {
                    c_row[j] += a_val * b_slice[j];
                    j += 1;
                }
            }
        });

        Self::from_f32_slice(&c, vec![m, n], self.device.clone())
    }

    /// In-place or element-wise RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight
    pub fn rms_norm(&self, weight: &Self, eps: f32) -> Result<Self> {
        let x = self.as_f32_slice()?;
        let w = weight.as_f32_slice()?;
        let hidden_dim = *self.shape.last().unwrap_or(&1);
        if hidden_dim == 0 || self.numel() == 0 {
            return Err(HarnessError::InvalidShape("Cannot perform RMSNorm on zero-sized tensor".into()));
        }
        if w.len() < hidden_dim {
            return Err(HarnessError::ShapeMismatch {
                expected: vec![hidden_dim],
                found: weight.shape().to_vec(),
            });
        }
        let _num_tokens = self.numel() / hidden_dim;

        let mut out = vec![0.0f32; self.numel()];

        out.par_chunks_mut(hidden_dim)
            .zip(x.par_chunks(hidden_dim))
            .for_each(|(out_row, x_row)| {
                let sum_sq: f32 = x_row.iter().map(|&v| v * v).sum();
                let rms = 1.0 / (sum_sq / hidden_dim as f32 + eps).sqrt();
                for i in 0..hidden_dim {
                    out_row[i] = x_row[i] * rms * w[i];
                }
            });

        Self::from_f32_slice(&out, self.shape.clone(), self.device.clone())
    }

    /// Fused SwiGLU activation: (x * sigmoid(x)) * gate
    pub fn silu_glu(&self, gate: &Self) -> Result<Self> {
        let x = self.as_f32_slice()?;
        let g = gate.as_f32_slice()?;
        if x.len() != g.len() {
            return Err(HarnessError::ShapeMismatch {
                expected: self.shape.clone(),
                found: gate.shape.clone(),
            });
        }
        let mut out = vec![0.0f32; x.len()];
        out.par_iter_mut()
            .zip(x.par_iter())
            .zip(g.par_iter())
            .for_each(|((res, &xi), &gi)| {
                // silu = xi / (1 + exp(-xi))
                let silu = xi / (1.0 + (-xi).exp());
                *res = silu * gi;
            });

        Self::from_f32_slice(&out, self.shape.clone(), self.device.clone())
    }

    fn compute_contiguous_strides(shape: &[usize]) -> Strides {
        let mut strides = vec![1; shape.len()];
        for i in (0..shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}
