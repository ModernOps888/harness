use byteorder::{LittleEndian, ReadBytesExt};
use harness_core::{
    HarnessError, MoEConfig, ModelArchitecture, ModelConfig, QuantizationMode, Result,
};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Cursor, Read};
use std::path::Path;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in Little Endian
pub const GGUF_DEFAULT_ALIGNMENT: u64 = 32;

/// Descriptor for an individual tensor in the GGUF tensor index
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub ggml_type: u32,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total number of elements in the tensor
    pub fn num_elements(&self) -> u64 {
        if self.dimensions.is_empty() {
            0
        } else {
            self.dimensions.iter().copied().product()
        }
    }

    /// Estimate the byte size of this tensor based on its GGML quantization block format
    pub fn byte_size(&self) -> usize {
        let elems = self.num_elements() as usize;
        let (block_elems, block_bytes) = ggml_type_block_geometry(self.ggml_type);
        if block_elems == 0 {
            return elems * 4;
        }
        (elems / block_elems) * block_bytes
    }
}

/// Returns (elements_per_block, bytes_per_block) for standard GGML tensor types
pub fn ggml_type_block_geometry(ggml_type: u32) -> (usize, usize) {
    match ggml_type {
        0 => (1, 4),     // F32
        1 => (1, 2),     // F16
        2 => (32, 18),   // Q4_0
        3 => (32, 20),   // Q4_1
        6 => (32, 22),   // Q5_0
        7 => (32, 34),   // Q8_0
        8 => (32, 36),   // Q8_1
        10 => (256, 84), // Q2_K
        11 => (256, 110),// Q3_K
        12 => (256, 144),// Q4_K
        13 => (256, 176),// Q5_K
        14 => (256, 210),// Q6_K
        15 => (256, 292),// Q8_K
        30 => (1, 2),    // BF16
        _ => (1, 2),
    }
}

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata_kv: HashMap<String, String>,
    pub tensors: Vec<GgufTensorInfo>,
    pub data_offset: u64,
}

pub type GgufHeader = GgufMetadata;

impl GgufMetadata {
    /// Extract a strongly-typed `ModelConfig` from parsed GGUF architectural metadata
    pub fn to_model_config(&self) -> ModelConfig {
        let arch_str = self
            .metadata_kv
            .get("general.architecture")
            .map(|s| s.as_str())
            .unwrap_or("llama");

        let arch = match arch_str {
            "qwen2" => ModelArchitecture::Qwen2_5,
            "qwen3" | "qwen3moe" => ModelArchitecture::Qwen3,
            "deepseek2" | "deepseek_v3" => ModelArchitecture::DeepSeekV3,
            "deepseek" | "deepseek_v4" => ModelArchitecture::DeepSeekV4,
            "phi3" | "phi4" => ModelArchitecture::Phi4,
            "gemma" | "gemma2" | "gemma3" => ModelArchitecture::Gemma3,
            "mistral" => ModelArchitecture::Mistral,
            _ => ModelArchitecture::Llama3,
        };

        let get_usize = |suffix: &str, default: usize| -> usize {
            let key = format!("{}.{}", arch_str, suffix);
            self.metadata_kv
                .get(&key)
                .and_then(|v| v.parse::<u64>().ok())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let get_f32 = |suffix: &str, default: f32| -> f32 {
            let key = format!("{}.{}", arch_str, suffix);
            self.metadata_kv
                .get(&key)
                .and_then(|v| v.parse::<f64>().ok())
                .map(|v| v as f32)
                .unwrap_or(default)
        };

        let num_hidden_layers = get_usize("block_count", 32);
        let hidden_size = get_usize("embedding_length", 4096);
        let intermediate_size = get_usize("feed_forward_length", 11008);
        let num_attention_heads = get_usize("attention.head_count", 32);
        let num_key_value_heads = get_usize("attention.head_count_kv", num_attention_heads);
        let max_position_embeddings = get_usize("context_length", 32768);
        let rms_norm_eps = get_f32("attention.layer_norm_rms_epsilon", 1e-6);
        let rope_theta = get_f32("rope.freq_base", 1_000_000.0);

        // Determine vocabulary size from arch.vocab_size, tokenizer token count, or token_embd.weight
        let vocab_size = self
            .metadata_kv
            .get(&format!("{}.vocab_size", arch_str))
            .or_else(|| self.metadata_kv.get("tokenizer.ggml.tokens.count"))
            .and_then(|v| v.parse::<usize>().ok())
            .or_else(|| {
                self.tensors
                    .iter()
                    .find(|t| t.name == "token_embd.weight")
                    .and_then(|t| t.dimensions.get(1).copied().map(|d| d as usize))
            })
            .unwrap_or(152064);

        // Check for Sparse MoE metadata (e.g. Qwen3-30B-A3B or DeepSeek-Coder-V2)
        let expert_count = get_usize("expert_count", 0);
        let expert_used_count = get_usize("expert_used_count", 0);
        let moe = if expert_count > 0 {
            let shared = get_usize("expert_shared_count", 0);
            let active = if expert_used_count > 0 { expert_used_count } else { 2 };
            Some(MoEConfig {
                num_routed_experts: expert_count,
                num_shared_experts: shared,
                num_active_experts: active,
                routing_top_k: active,
                norm_topk_prob: true,
            })
        } else {
            None
        };

        let quantization = match self
            .metadata_kv
            .get("general.file_type")
            .and_then(|v| v.parse::<u32>().ok())
        {
            Some(14) | Some(15) => QuantizationMode::Q4_K_M,
            Some(10) | Some(11) | Some(12) => QuantizationMode::Q3_K_M,
            Some(7) => QuantizationMode::INT8,
            _ => QuantizationMode::Q4_K_M,
        };

        ModelConfig {
            architecture: arch,
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            max_position_embeddings,
            rms_norm_eps,
            rope_theta,
            moe,
            quantization,
        }
    }
}

pub struct GgufReader;

impl GgufReader {
    /// Memory-map and inspect a GGUF file on disk without loading multi-GB tensor weights into RAM
    pub fn inspect_file<P: AsRef<Path>>(path: P) -> Result<GgufMetadata> {
        let p = path.as_ref();
        let file = File::open(p).map_err(|e| {
            HarnessError::ModelLoad(format!("Failed to open GGUF file {:?}: {}", p, e))
        })?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::parse_header(&mmap)
    }

    /// Extract and dequantize up to `max_elements` of a named tensor (`F32`, `F16`, `Q8_0`, `Q4_K`)
    /// directly from a memory-mapped GGUF file on disk.
    pub fn extract_tensor_f32<P: AsRef<Path>>(
        path: P,
        tensor_name: &str,
        max_elements: usize,
    ) -> Result<(GgufTensorInfo, Vec<f32>)> {
        let p = path.as_ref();
        let file = File::open(p).map_err(|e| {
            HarnessError::ModelLoad(format!("Failed to open GGUF file {:?}: {}", p, e))
        })?;
        let mmap = unsafe { Mmap::map(&file)? };
        let meta = Self::parse_header(&mmap)?;
        let info = meta
            .tensors
            .iter()
            .find(|t| t.name == tensor_name)
            .cloned()
            .ok_or_else(|| {
                HarnessError::ModelLoad(format!("Tensor '{}' not found in GGUF", tensor_name))
            })?;

        let start = (meta.data_offset + info.offset) as usize;
        let byte_len = info.byte_size();
        if start.saturating_add(byte_len) > mmap.len() {
            return Err(HarnessError::ModelLoad(format!(
                "Tensor '{}' slice [{}..{}] exceeds GGUF file length {}",
                tensor_name,
                start,
                start.saturating_add(byte_len),
                mmap.len()
            )));
        }
        let raw = &mmap[start..start + byte_len];
        let total_elems = (info.num_elements() as usize).min(max_elements);

        let values = match info.ggml_type {
            0 => {
                // F32
                let count = total_elems.min(raw.len() / 4);
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let b = [raw[i * 4], raw[i * 4 + 1], raw[i * 4 + 2], raw[i * 4 + 3]];
                    out.push(f32::from_le_bytes(b));
                }
                out
            }
            1 => {
                // F16
                let count = total_elems.min(raw.len() / 2);
                let mut out = Vec::with_capacity(count);
                for i in 0..count {
                    let h = u16::from_le_bytes([raw[i * 2], raw[i * 2 + 1]]);
                    out.push(harness_quant::f16_to_f32(h));
                }
                out
            }
            7 => {
                // Q8_0 (32 elements per 34-byte block)
                let max_blocks = total_elems.div_ceil(32);
                let mut out = harness_quant::dequantize_q8_0_bytes(raw, max_blocks);
                out.truncate(total_elems);
                out
            }
            12 => {
                // Q4_K (256 elements per 144-byte super-block)
                let max_blocks = total_elems.div_ceil(256);
                let mut out = harness_quant::dequantize_q4_k_bytes(raw, max_blocks);
                out.truncate(total_elems);
                out
            }
            other => {
                return Err(HarnessError::ModelLoad(format!(
                    "Unsupported GGML tensor type {} for direct extraction on '{}'",
                    other, tensor_name
                )))
            }
        };

        Ok((info, values))
    }

    /// Parse GGUF v2/v3 header, full metadata KV table (all 13 types), and tensor descriptor table
    pub fn parse_header(bytes: &[u8]) -> Result<GgufMetadata> {
        if bytes.len() < 24 {
            return Err(HarnessError::ModelLoad("GGUF header too small".into()));
        }
        let magic = u32::from_le_bytes(bytes[0..4].try_into().unwrap());
        if magic != GGUF_MAGIC {
            return Err(HarnessError::ModelLoad(format!(
                "Invalid GGUF magic 0x{:08X}, expected 0x{:08X}",
                magic, GGUF_MAGIC
            )));
        }
        let version = u32::from_le_bytes(bytes[4..8].try_into().unwrap());
        let tensor_count = u64::from_le_bytes(bytes[8..16].try_into().unwrap());
        let kv_count = u64::from_le_bytes(bytes[16..24].try_into().unwrap());

        let mut metadata_kv = HashMap::new();
        let mut tensors = Vec::new();
        let mut cursor = Cursor::new(bytes);
        cursor.set_position(24);

        // Parse KV metadata section if bytes extend beyond the 24-byte preamble
        let mut parsed_all_kvs = true;
        for _ in 0..kv_count {
            if cursor.position() >= bytes.len() as u64 {
                parsed_all_kvs = false;
                break;
            }
            let key = match Self::read_gguf_string(&mut cursor) {
                Ok(k) => k,
                Err(_) => {
                    parsed_all_kvs = false;
                    break;
                }
            };
            let val_type = match cursor.read_u32::<LittleEndian>() {
                Ok(vt) => vt,
                Err(_) => {
                    parsed_all_kvs = false;
                    break;
                }
            };
            match Self::read_gguf_value(&mut cursor, val_type, &key, &mut metadata_kv) {
                Ok(val_str) => {
                    metadata_kv.insert(key, val_str);
                }
                Err(_) => {
                    parsed_all_kvs = false;
                    break;
                }
            }
        }

        // Parse Tensor Info descriptor table if all KVs were present in the buffer
        if parsed_all_kvs {
            for _ in 0..tensor_count {
                if cursor.position() >= bytes.len() as u64 {
                    break;
                }
                let name = match Self::read_gguf_string(&mut cursor) {
                    Ok(n) => n,
                    Err(_) => break,
                };
                let n_dims = match cursor.read_u32::<LittleEndian>() {
                    Ok(d) => d as usize,
                    Err(_) => break,
                };
                let mut dimensions = Vec::with_capacity(n_dims);
                let mut dim_ok = true;
                for _ in 0..n_dims {
                    match cursor.read_u64::<LittleEndian>() {
                        Ok(d) => dimensions.push(d),
                        Err(_) => {
                            dim_ok = false;
                            break;
                        }
                    }
                }
                if !dim_ok {
                    break;
                }
                let ggml_type = match cursor.read_u32::<LittleEndian>() {
                    Ok(t) => t,
                    Err(_) => break,
                };
                let offset = match cursor.read_u64::<LittleEndian>() {
                    Ok(o) => o,
                    Err(_) => break,
                };
                tensors.push(GgufTensorInfo {
                    name,
                    dimensions,
                    ggml_type,
                    offset,
                });
            }
        }

        let alignment = metadata_kv
            .get("general.alignment")
            .and_then(|v| v.parse::<u64>().ok())
            .unwrap_or(GGUF_DEFAULT_ALIGNMENT)
            .max(1);

        let pos = cursor.position();
        let data_offset = pos.div_ceil(alignment) * alignment;

        Ok(GgufMetadata {
            version,
            tensor_count,
            kv_count,
            metadata_kv,
            tensors,
            data_offset,
        })
    }

    fn read_gguf_string(cursor: &mut Cursor<&[u8]>) -> Result<String> {
        let len = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| HarnessError::ModelLoad(format!("GGUF string len EOF: {}", e)))?
            as usize;
        let pos = cursor.position() as usize;
        let slice = cursor.get_ref();
        if pos.saturating_add(len) > slice.len() {
            return Err(HarnessError::ModelLoad("GGUF string exceeds buffer".into()));
        }
        let s = String::from_utf8_lossy(&slice[pos..pos + len]).into_owned();
        cursor.set_position((pos + len) as u64);
        Ok(s)
    }

    fn skip_gguf_string(cursor: &mut Cursor<&[u8]>) -> Result<()> {
        let len = cursor
            .read_u64::<LittleEndian>()
            .map_err(|e| HarnessError::ModelLoad(format!("GGUF string skip EOF: {}", e)))?
            as usize;
        let pos = cursor.position() as usize;
        let slice = cursor.get_ref();
        if pos.saturating_add(len) > slice.len() {
            return Err(HarnessError::ModelLoad("GGUF string skip exceeds buffer".into()));
        }
        cursor.set_position((pos + len) as u64);
        Ok(())
    }

    fn read_gguf_value(
        cursor: &mut Cursor<&[u8]>,
        val_type: u32,
        key: &str,
        extra_kv: &mut HashMap<String, String>,
    ) -> Result<String> {
        match val_type {
            0 => Ok(cursor.read_u8()?.to_string()),
            1 => Ok(cursor.read_i8()?.to_string()),
            2 => Ok(cursor.read_u16::<LittleEndian>()?.to_string()),
            3 => Ok(cursor.read_i16::<LittleEndian>()?.to_string()),
            4 => Ok(cursor.read_u32::<LittleEndian>()?.to_string()),
            5 => Ok(cursor.read_i32::<LittleEndian>()?.to_string()),
            6 => Ok(cursor.read_f32::<LittleEndian>()?.to_string()),
            7 => Ok((cursor.read_u8()? != 0).to_string()),
            8 => Self::read_gguf_string(cursor),
            9 => {
                // GGUF_METADATA_VALUE_TYPE_ARRAY
                let elem_type = cursor.read_u32::<LittleEndian>()?;
                let array_len = cursor.read_u64::<LittleEndian>()? as usize;
                extra_kv.insert(format!("{}.count", key), array_len.to_string());

                // Fast-skip large arrays (such as 152,064 tokenizer entries)
                if array_len > 32 {
                    Self::skip_gguf_array(cursor, elem_type, array_len)?;
                    Ok(format!("<array len={}>", array_len))
                } else {
                    let mut items = Vec::with_capacity(array_len);
                    for _ in 0..array_len {
                        items.push(Self::read_gguf_value(cursor, elem_type, key, extra_kv)?);
                    }
                    Ok(format!("[{}]", items.join(", ")))
                }
            }
            10 => Ok(cursor.read_u64::<LittleEndian>()?.to_string()),
            11 => Ok(cursor.read_i64::<LittleEndian>()?.to_string()),
            12 => Ok(cursor.read_f64::<LittleEndian>()?.to_string()),
            other => Err(HarnessError::ModelLoad(format!(
                "Unsupported GGUF metadata value type {}",
                other
            ))),
        }
    }

    fn skip_gguf_array(
        cursor: &mut Cursor<&[u8]>,
        elem_type: u32,
        array_len: usize,
    ) -> Result<()> {
        let fixed_elem_size: Option<usize> = match elem_type {
            0 | 1 | 7 => Some(1),
            2 | 3 => Some(2),
            4..=6 => Some(4),
            10..=12 => Some(8),
            _ => None,
        };

        if let Some(elem_bytes) = fixed_elem_size {
            let total_skip = array_len.saturating_mul(elem_bytes);
            let pos = cursor.position() as usize;
            if pos.saturating_add(total_skip) > cursor.get_ref().len() {
                return Err(HarnessError::ModelLoad("GGUF array skip exceeds buffer".into()));
            }
            cursor.set_position((pos + total_skip) as u64);
            return Ok(());
        }

        for _ in 0..array_len {
            if elem_type == 8 {
                Self::skip_gguf_string(cursor)?;
            } else if elem_type == 9 {
                let nested_type = cursor.read_u32::<LittleEndian>()?;
                let nested_len = cursor.read_u64::<LittleEndian>()? as usize;
                Self::skip_gguf_array(cursor, nested_type, nested_len)?;
            } else {
                let mut dummy = [0u8; 1];
                cursor.read_exact(&mut dummy)?;
            }
        }
        Ok(())
    }
}

