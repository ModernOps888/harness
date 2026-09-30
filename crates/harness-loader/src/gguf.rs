use harness_core::{HarnessError, Result};
use std::collections::HashMap;

pub const GGUF_MAGIC: u32 = 0x46554747; // "GGUF" in Little Endian

#[derive(Debug, Clone)]
pub struct GgufMetadata {
    pub version: u32,
    pub tensor_count: u64,
    pub kv_count: u64,
    pub metadata_kv: HashMap<String, String>,
}

pub struct GgufReader;

impl GgufReader {
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

        Ok(GgufMetadata {
            version,
            tensor_count,
            kv_count,
            metadata_kv: HashMap::new(),
        })
    }
}
