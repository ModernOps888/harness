use harness_core::{HarnessError, Result};
use std::collections::HashMap;

pub struct SafeTensorsReader;

impl SafeTensorsReader {
    pub fn parse_header(bytes: &[u8]) -> Result<HashMap<String, serde_json::Value>> {
        if bytes.len() < 8 {
            return Err(HarnessError::ModelLoad("Invalid SafeTensors header length".into()));
        }
        let header_len = u64::from_le_bytes(bytes[0..8].try_into().unwrap()) as usize;
        if bytes.len() < 8 + header_len {
            return Err(HarnessError::ModelLoad("Truncated SafeTensors file".into()));
        }
        let header_json: HashMap<String, serde_json::Value> =
            serde_json::from_slice(&bytes[8..8 + header_len])?;
        Ok(header_json)
    }
}
