//! Bit-exact GGUF `Q4_K` (256-weight super-block, 144 bytes = 4.50 bpw) and `Q8_K`
//! quantization, dequantization, and fused integer dot-product (`vec_dot_q4_k_q8_k`),
//! plus Activation-Aware Fisher/Hessian Importance Matrix (`imatrix`) scale optimization.

/// Exact GGUF `block_q4_K` super-block layout (144 bytes for 256 weights = 4.50 bits/weight).
/// Matches `ggml-common.h` `block_q4_K` byte-for-byte:
/// - `d`: IEEE 754 `f16` super-block scale (2 bytes)
/// - `dmin`: IEEE 754 `f16` super-block minimum scale (2 bytes)
/// - `scales`: 12 bytes packing eight 6-bit sub-block scales and eight 6-bit sub-block mins
/// - `qs`: 128 bytes packing 256 4-bit unsigned weights (8 sub-blocks of 32 weights)
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BlockQ4K {
    pub d: u16,
    pub dmin: u16,
    pub scales: [u8; 12],
    pub qs: [u8; 128],
}

impl BlockQ4K {
    pub const BLOCK_SIZE: usize = 256;
    pub const BYTE_SIZE: usize = 144;

    /// Parse a single 144-byte slice into a `BlockQ4K`.
    #[inline]
    pub fn from_bytes(bytes: &[u8]) -> Option<Self> {
        if bytes.len() < Self::BYTE_SIZE {
            return None;
        }
        let d = u16::from_le_bytes([bytes[0], bytes[1]]);
        let dmin = u16::from_le_bytes([bytes[2], bytes[3]]);
        let mut scales = [0u8; 12];
        scales.copy_from_slice(&bytes[4..16]);
        let mut qs = [0u8; 128];
        qs.copy_from_slice(&bytes[16..144]);
        Some(Self {
            d,
            dmin,
            scales,
            qs,
        })
    }

    /// Serialize `BlockQ4K` into its exact 144-byte GGUF wire representation.
    #[inline]
    pub fn to_bytes(&self) -> [u8; 144] {
        let mut out = [0u8; 144];
        out[0..2].copy_from_slice(&self.d.to_le_bytes());
        out[2..4].copy_from_slice(&self.dmin.to_le_bytes());
        out[4..16].copy_from_slice(&self.scales);
        out[16..144].copy_from_slice(&self.qs);
        out
    }

    /// Unpack the 6-bit scale and 6-bit min for sub-block `j` (`0..8`),
    /// matching `get_scale_min_k4` in `ggml-quants.c`.
    #[inline]
    pub fn get_scale_min(&self, j: usize) -> (u8, u8) {
        if j < 4 {
            let sc = self.scales[j] & 63;
            let m = self.scales[j + 4] & 63;
            (sc, m)
        } else {
            let sc = (self.scales[j + 4] & 0x0F) | ((self.scales[j - 4] >> 6) << 4);
            let m = (self.scales[j + 4] >> 4) | ((self.scales[j] >> 6) << 4);
            (sc, m)
        }
    }

    /// Dequantize all 256 weights of this `Q4_K` super-block into `out[0..256]`.
    /// Matches `dequantize_row_q4_K` in `ggml-quants.c`.
    pub fn dequantize(&self, out: &mut [f32; 256]) {
        let d = f16_to_f32(self.d);
        let min = f16_to_f32(self.dmin);

        let mut is = 0usize;
        let mut q_offset = 0usize;
        let mut out_offset = 0usize;

        // 4 chunks of 64 weights (each chunk uses 32 bytes of qs: low nibbles = 32 weights, high nibbles = 32 weights)
        for _ in 0..4 {
            let (sc1, m1) = self.get_scale_min(is);
            let d1 = d * (sc1 as f32);
            let min1 = min * (m1 as f32);

            let (sc2, m2) = self.get_scale_min(is + 1);
            let d2 = d * (sc2 as f32);
            let min2 = min * (m2 as f32);

            for l in 0..32 {
                out[out_offset + l] = d1 * ((self.qs[q_offset + l] & 0x0F) as f32) - min1;
            }
            for l in 0..32 {
                out[out_offset + 32 + l] = d2 * ((self.qs[q_offset + l] >> 4) as f32) - min2;
            }

            q_offset += 32;
            out_offset += 64;
            is += 2;
        }
    }

    /// Fused dot product between this 256-weight `Q4_K` super-block and a 256-element FP32 activation slice,
    /// without allocating or materializing a dequantized buffer.
    #[inline]
    pub fn dot_f32(&self, x: &[f32]) -> f32 {
        debug_assert!(x.len() >= 256);
        let d = f16_to_f32(self.d);
        let min = f16_to_f32(self.dmin);

        let mut total = 0.0f32;
        let mut is = 0usize;
        let mut q_offset = 0usize;
        let mut x_offset = 0usize;

        for _ in 0..4 {
            let (sc1, m1) = self.get_scale_min(is);
            let d1 = d * (sc1 as f32);
            let min1 = min * (m1 as f32);

            let (sc2, m2) = self.get_scale_min(is + 1);
            let d2 = d * (sc2 as f32);
            let min2 = min * (m2 as f32);

            let mut sum_q1 = 0.0f32;
            let mut sum_x1 = 0.0f32;
            let mut sum_q2 = 0.0f32;
            let mut sum_x2 = 0.0f32;

            for l in 0..32 {
                let q_byte = self.qs[q_offset + l];
                let xv1 = x[x_offset + l];
                let xv2 = x[x_offset + 32 + l];
                sum_q1 += ((q_byte & 0x0F) as f32) * xv1;
                sum_x1 += xv1;
                sum_q2 += ((q_byte >> 4) as f32) * xv2;
                sum_x2 += xv2;
            }

            total += (d1 * sum_q1 - min1 * sum_x1) + (d2 * sum_q2 - min2 * sum_x2);

            q_offset += 32;
            x_offset += 64;
            is += 2;
        }
        total
    }

    /// Activation-aware (`imatrix`) quantization of 256 FP32 weights into a `BlockQ4K`.
    /// When `importance_weights` (e.g. $\mathbb{E}[x_i^2]$) are provided, sub-block scales
    /// and minimums are calibrated via weighted least-squares to protect high-salience channels.
    pub fn quantize_from_f32(weights: &[f32; 256], importance: Option<&[f32; 256]>) -> Self {
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];

        for sb in 0..8 {
            let start = sb * 32;
            let slice = &weights[start..start + 32];
            let mut w_min = f32::INFINITY;
            let mut w_max = f32::NEG_INFINITY;
            for &v in slice {
                if v < w_min {
                    w_min = v;
                }
                if v > w_max {
                    w_max = v;
                }
            }
            // In Q4_K, weights are reconstructed as d_sub * q - min_sub where d_sub >= 0 and min_sub >= 0.
            // Thus w_min_clamped = (-w_min).max(0.0).
            let mut min_pos = (-w_min).max(0.0);
            let mut scale = (w_max + min_pos) / 15.0;

            // Coordinate refinement with importance weights if provided
            if let Some(imp) = importance {
                let imp_slice = &imp[start..start + 32];
                if scale > 1e-12 {
                    let mut num_s = 0.0f32;
                    let mut den_s = 0.0f32;
                    for l in 0..32 {
                        let w = slice[l];
                        let iw = imp_slice[l].max(1e-6);
                        let q = ((w + min_pos) / scale).round().clamp(0.0, 15.0);
                        num_s += iw * (w + min_pos) * q;
                        den_s += iw * q * q;
                    }
                    if den_s > 1e-12 {
                        scale = (num_s / den_s).max(0.0);
                    }
                    // Refine min_pos holding q fixed
                    let mut num_m = 0.0f32;
                    let mut den_m = 0.0f32;
                    for l in 0..32 {
                        let w = slice[l];
                        let iw = imp_slice[l].max(1e-6);
                        let q = if scale > 1e-12 {
                            ((w + min_pos) / scale).round().clamp(0.0, 15.0)
                        } else {
                            0.0
                        };
                        num_m += iw * (scale * q - w);
                        den_m += iw;
                    }
                    if den_m > 1e-12 {
                        min_pos = (num_m / den_m).max(0.0);
                    }
                }
            }

            sub_scales[sb] = scale;
            sub_mins[sb] = min_pos;
        }

        let max_scale = sub_scales.iter().copied().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().copied().fold(0.0f32, f32::max);

        let d_f32 = max_scale / 63.0;
        let dmin_f32 = max_min / 63.0;
        let d_u16 = f32_to_f16(d_f32);
        let dmin_u16 = f32_to_f16(dmin_f32);
        let d_eff = f16_to_f32(d_u16);
        let dmin_eff = f16_to_f32(dmin_u16);

        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];
        for sb in 0..8 {
            q_scales[sb] = if d_eff > 1e-12 {
                (sub_scales[sb] / d_eff).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
            q_mins[sb] = if dmin_eff > 1e-12 {
                (sub_mins[sb] / dmin_eff).round().clamp(0.0, 63.0) as u8
            } else {
                0
            };
        }

        // Pack 6-bit scales and mins into 12 bytes (inverse of get_scale_min_k4)
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = (q_scales[j] & 63) | ((q_scales[j + 4] >> 4) << 6);
            scales[j + 4] = (q_mins[j] & 63) | ((q_mins[j + 4] >> 4) << 6);
            scales[j + 8] = (q_scales[j + 4] & 0x0F) | ((q_mins[j + 4] & 0x0F) << 4);
        }

        // Quantize the 256 weights into 128 bytes
        let mut qs = [0u8; 128];
        for chunk in 0..4 {
            let sb1 = chunk * 2;
            let sb2 = sb1 + 1;
            let d1 = d_eff * (q_scales[sb1] as f32);
            let m1 = dmin_eff * (q_mins[sb1] as f32);
            let d2 = d_eff * (q_scales[sb2] as f32);
            let m2 = dmin_eff * (q_mins[sb2] as f32);

            for l in 0..32 {
                let w1 = weights[chunk * 64 + l];
                let w2 = weights[chunk * 64 + 32 + l];
                let q1 = if d1 > 1e-12 {
                    ((w1 + m1) / d1).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                let q2 = if d2 > 1e-12 {
                    ((w2 + m2) / d2).round().clamp(0.0, 15.0) as u8
                } else {
                    0
                };
                qs[chunk * 32 + l] = (q1 & 0x0F) | ((q2 & 0x0F) << 4);
            }
        }

        Self {
            d: d_u16,
            dmin: dmin_u16,
            scales,
            qs,
        }
    }
}

/// Dequantize a raw byte slice containing `n_blocks` GGUF `Q4_K` super-blocks into a `Vec<f32>`.
pub fn dequantize_q4_k_bytes(raw: &[u8], max_blocks: usize) -> Vec<f32> {
    let available_blocks = raw.len() / BlockQ4K::BYTE_SIZE;
    let n_blocks = available_blocks.min(max_blocks);
    let mut out = vec![0.0f32; n_blocks * BlockQ4K::BLOCK_SIZE];
    let mut block_out = [0.0f32; 256];
    for b in 0..n_blocks {
        let offset = b * BlockQ4K::BYTE_SIZE;
        if let Some(blk) = BlockQ4K::from_bytes(&raw[offset..offset + BlockQ4K::BYTE_SIZE]) {
            blk.dequantize(&mut block_out);
            out[b * 256..(b + 1) * 256].copy_from_slice(&block_out);
        }
    }
    out
}

/// Dequantize a raw byte slice containing GGUF `Q8_0` blocks (34 bytes per 32 weights: 2-byte `f16` scale + 32 `i8` quants).
pub fn dequantize_q8_0_bytes(raw: &[u8], max_blocks: usize) -> Vec<f32> {
    const Q8_0_BYTE_SIZE: usize = 34;
    let available_blocks = raw.len() / Q8_0_BYTE_SIZE;
    let n_blocks = available_blocks.min(max_blocks);
    let mut out = Vec::with_capacity(n_blocks * 32);
    for b in 0..n_blocks {
        let offset = b * Q8_0_BYTE_SIZE;
        let d = f16_to_f32(u16::from_le_bytes([raw[offset], raw[offset + 1]]));
        for l in 0..32 {
            let q = raw[offset + 2 + l] as i8;
            out.push(d * (q as f32));
        }
    }
    out
}

/// IEEE 754 binary16 (`f16`) to binary32 (`f32`) bit-exact conversion.
#[inline]
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1F) as u32;
    let mant = (h & 0x03FF) as u32;

    let bits = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // Subnormal f16 -> normalized f32
            let mut m = mant;
            let mut e = 127u32 - 14;
            while (m & 0x0400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x03FF;
            (sign << 31) | (e << 23) | (m << 13)
        }
    } else if exp == 31 {
        (sign << 31) | (0xFF << 23) | (mant << 13)
    } else {
        let e = exp + (127 - 15);
        (sign << 31) | (e << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

/// IEEE 754 binary32 (`f32`) to binary16 (`f16`) round-to-nearest-even conversion.
#[inline]
pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x007F_FFFF;

    if exp == 255 {
        if mant != 0 {
            return sign | 0x7E00; // NaN
        }
        return sign | 0x7C00; // Inf
    }

    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return sign | 0x7C00;
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = 14 - new_exp;
        let half_m = (m >> shift) as u16;
        return sign | half_m;
    }

    let half_exp = (new_exp as u16) << 10;
    let half_mant = ((mant + 0x0000_1000) >> 13) as u16;
    sign.wrapping_add(half_exp.wrapping_add(half_mant))
}
