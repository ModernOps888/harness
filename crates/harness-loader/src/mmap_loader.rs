use byteorder::{LittleEndian, ReadBytesExt};
use harness_core::{DType, HarnessError, Result, Shape, Tensor};
use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::io::Cursor;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::info;

/// Metadata for an individual tensor within a memory-mapped weight file
#[derive(Debug, Clone)]
pub struct LoadedWeight {
    pub name: String,
    pub shape: Shape,
    pub dtype: DType,
    pub byte_offset: usize,
    pub byte_len: usize,
}

/// Zero-copy memory mapped model file loader
pub struct MmapModelLoader {
    path: PathBuf,
    mmap: Arc<Mmap>,
    weights: HashMap<String, LoadedWeight>,
}

impl MmapModelLoader {
    /// Open a file and memory-map it directly to virtual address space
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let p = path.as_ref();
        if !p.exists() || !p.is_file() {
            return Err(HarnessError::ModelLoad(format!(
                "Model weight path does not exist or is not a valid file: {:?}",
                p
            )));
        }
        let canonical_path = p.canonicalize()?;
        let file = File::open(&canonical_path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        info!(path = ?canonical_path, size_bytes = mmap.len(), "Memory-mapped model weights file");

        let mut loader = Self {
            path: canonical_path,
            mmap: Arc::new(mmap),
            weights: HashMap::new(),
        };

        loader.index_weights()?;
        Ok(loader)
    }

    fn index_weights(&mut self) -> Result<()> {
        if self.mmap.len() < 8 {
            return Err(HarnessError::ModelLoad("File too small for SafeTensors or GGUF".into()));
        }

        // Check if SafeTensors format: first 8 bytes is u64 little-endian header length
        let mut rdr = Cursor::new(&self.mmap[0..8]);
        let header_len = rdr.read_u64::<LittleEndian>().unwrap_or(0) as usize;

        if header_len > 0 && header_len + 8 <= self.mmap.len() {
            // SafeTensors format
            let header_bytes = &self.mmap[8..8 + header_len];
            if let Ok(header_json) = serde_json::from_slice::<serde_json::Value>(header_bytes) {
                if let Some(obj) = header_json.as_object() {
                    let data_start = 8 + header_len;
                    for (name, val) in obj {
                        if name == "__metadata__" {
                            continue;
                        }
                        let dtype_str = val.get("dtype").and_then(|v| v.as_str()).unwrap_or("F32");
                        let shape_vec: Vec<usize> = val
                            .get("shape")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
                            .unwrap_or_default();

                        let offsets: Vec<usize> = val
                            .get("data_offsets")
                            .and_then(|v| v.as_array())
                            .map(|arr| arr.iter().filter_map(|x| x.as_u64().map(|n| n as usize)).collect())
                            .unwrap_or_default();

                        if offsets.len() == 2 {
                            let dtype = match dtype_str {
                                "F32" => DType::F32,
                                "F16" => DType::F16,
                                "BF16" => DType::BF16,
                                "I8" => DType::I8,
                                _ => DType::F32,
                            };
                            let start = data_start + offsets[0];
                            let len = offsets[1] - offsets[0];

                            self.weights.insert(
                                name.clone(),
                                LoadedWeight {
                                    name: name.clone(),
                                    shape: shape_vec,
                                    dtype,
                                    byte_offset: start,
                                    byte_len: len,
                                },
                            );
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Read tensor bytes as zero-copy slice and construct a Tensor
    pub fn get_tensor(&self, name: &str, device: &harness_core::Device) -> Result<Tensor> {
        let meta = self
            .weights
            .get(name)
            .ok_or_else(|| HarnessError::ModelLoad(format!("Tensor {} not found in weights", name)))?;

        let end = meta.byte_offset + meta.byte_len;
        if end > self.mmap.len() {
            return Err(HarnessError::ModelLoad("Weight offset exceeds file boundary".into()));
        }

        let slice = &self.mmap[meta.byte_offset..end];
        Tensor::from_raw_bytes(slice.to_vec(), meta.shape.clone(), meta.dtype, device.clone())
    }

    pub fn list_weight_names(&self) -> Vec<String> {
        self.weights.keys().cloned().collect()
    }

    pub fn total_weights_bytes(&self) -> usize {
        self.weights.values().map(|w| w.byte_len).sum()
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}
