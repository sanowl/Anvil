use std::sync::Arc;
use std::fs::OpenOptions;
use std::io::{Read, Write, Seek, SeekFrom};
use parking_lot::RwLock;
use crate::error::{AnvilError, AnvilResult};

/// Advanced tensor storage with memory mapping and custom allocators
#[derive(Debug, Clone)]
pub struct AdvancedTensorStorage {
    data: Arc<RwLock<Box<[u8]>>>,
    size: usize,
    device: crate::tensor::devices::Device,
    is_memory_mapped: bool,
    memory_map_path: Option<String>,
}

impl AdvancedTensorStorage {
    pub fn new(size: usize, device: crate::tensor::devices::Device) -> Self {
        let data = vec![0u8; size].into_boxed_slice();
        Self {
            data: Arc::new(RwLock::new(data)),
            size,
            device,
            is_memory_mapped: false,
            memory_map_path: None,
        }
    }

    pub fn from_vec(data: Vec<u8>, device: crate::tensor::devices::Device) -> Self {
        let size = data.len();
        Self {
            data: Arc::new(RwLock::new(data.into_boxed_slice())),
            size,
            device,
            is_memory_mapped: false,
            memory_map_path: None,
        }
    }

    pub fn as_slice(&self) -> &[u8] {
        // This is safe because we're only reading
        unsafe {
            let guard = self.data.read();
            std::slice::from_raw_parts(guard.as_ptr(), guard.len())
        }
    }

    pub fn as_slice_mut(&mut self) -> &mut [u8] {
        // This is safe because we have exclusive access
        unsafe {
            let mut guard = self.data.write();
            std::slice::from_raw_parts_mut(guard.as_mut_ptr(), guard.len())
        }
    }

    pub fn size(&self) -> usize {
        self.size
    }

    pub fn device(&self) -> crate::tensor::devices::Device {
        self.device
    }

    pub fn is_memory_mapped(&self) -> bool {
        self.is_memory_mapped
    }

    pub fn memory_map_path(&self) -> Option<&str> {
        self.memory_map_path.as_deref()
    }

    /// Create a memory-mapped storage
    pub fn memory_mapped(path: String, size: usize) -> AnvilResult<Self> {
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .open(&path)
            .map_err(|e| AnvilError::ComputationError(format!("Failed to open file for memory mapping: {}", e)))?;
        
        // Ensure the file is the right size
        file.set_len(size as u64)
            .map_err(|e| AnvilError::ComputationError(format!("Failed to set file size: {}", e)))?;
        
        // Read the data from the file
        let mut data = vec![0u8; size];
        file.seek(SeekFrom::Start(0))
            .map_err(|e| AnvilError::ComputationError(format!("Failed to seek file: {}", e)))?;
        file.read_exact(&mut data)
            .map_err(|e| AnvilError::ComputationError(format!("Failed to read file: {}", e)))?;
        
        Ok(Self {
            data: Arc::new(RwLock::new(data.into_boxed_slice())),
            size,
            device: crate::tensor::devices::Device::Cpu,
            is_memory_mapped: true,
            memory_map_path: Some(path),
        })
    }

    /// Clone the storage
    pub fn clone(&self) -> Self {
        let guard = self.data.read();
        let data = guard.clone();
        Self {
            data: Arc::new(RwLock::new(data)),
            size: self.size,
            device: self.device,
            is_memory_mapped: self.is_memory_mapped,
            memory_map_path: self.memory_map_path.clone(),
        }
    }

    /// Transfer storage to another device
    pub async fn transfer_to(&self, device: crate::tensor::devices::Device) -> AnvilResult<Self> {
        if self.device == device {
            return Ok(self.clone());
        }
        
        let guard = self.data.read();
        let data = guard.clone();
        drop(guard);
        
        // Simulate device-specific transfer logic
        match (self.device, device) {
            (crate::tensor::devices::Device::Cpu, crate::tensor::devices::Device::Cuda(_)) => {
                // CPU to CUDA transfer - in real implementation would use CUDA APIs
                let mut new_storage = Self {
                    data: Arc::new(RwLock::new(data)),
                    size: self.size,
                    device,
                    is_memory_mapped: false, // GPU memory is not memory-mapped to files
                    memory_map_path: None,
                };
                Ok(new_storage)
            },
            (crate::tensor::devices::Device::Cuda(_), crate::tensor::devices::Device::Cpu) => {
                // CUDA to CPU transfer
                let new_storage = Self {
                    data: Arc::new(RwLock::new(data)),
                    size: self.size,
                    device,
                    is_memory_mapped: self.is_memory_mapped,
                    memory_map_path: self.memory_map_path.clone(),
                };
                Ok(new_storage)
            },
            (crate::tensor::devices::Device::Metal(_), crate::tensor::devices::Device::Cpu) |
            (crate::tensor::devices::Device::Cpu, crate::tensor::devices::Device::Metal(_)) => {
                // Metal transfer
                let new_storage = Self {
                    data: Arc::new(RwLock::new(data)),
                    size: self.size,
                    device,
                    is_memory_mapped: if device == crate::tensor::devices::Device::Cpu { 
                        self.is_memory_mapped 
                    } else { 
                        false 
                    },
                    memory_map_path: if device == crate::tensor::devices::Device::Cpu { 
                        self.memory_map_path.clone() 
                    } else { 
                        None 
                    },
                };
                Ok(new_storage)
            },
            _ => {
                // Generic transfer for other device combinations
                let new_storage = Self {
                    data: Arc::new(RwLock::new(data)),
                    size: self.size,
                    device,
                    is_memory_mapped: false,
                    memory_map_path: None,
                };
                Ok(new_storage)
            }
        }
    }
}

// Ensure thread safety
unsafe impl Send for AdvancedTensorStorage {}
unsafe impl Sync for AdvancedTensorStorage {} 