//! Advanced device abstraction with async support

use serde::{Serialize, Deserialize};
use crate::error::{AnvilError, AnvilResult};

/// Advanced device abstraction with async support
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Device {
    Cpu,
    Cuda(usize),
    Metal(usize),
    Vulkan(usize),
    OpenCL(usize),
}

impl Device {
    /// Check if device supports async operations
    pub const fn supports_async(&self) -> bool {
        matches!(self, Device::Cuda(_) | Device::Metal(_) | Device::Vulkan(_) | Device::OpenCL(_))
    }

    /// Check if device is GPU-based
    pub const fn is_gpu(&self) -> bool {
        matches!(self, Device::Cuda(_) | Device::Metal(_) | Device::Vulkan(_) | Device::OpenCL(_))
    }

    /// Get device memory capacity (in bytes) - simplified for now
    pub async fn memory_capacity(&self) -> AnvilResult<usize> {
        match self {
            Device::Cpu => Ok(usize::MAX), // Simplified
            Device::Cuda(_) => Ok(8 * 1024 * 1024 * 1024), // 8GB default
            Device::Metal(_) => Ok(8 * 1024 * 1024 * 1024),
            Device::Vulkan(_) => Ok(4 * 1024 * 1024 * 1024),
            Device::OpenCL(_) => Ok(4 * 1024 * 1024 * 1024),
        }
    }

    /// Convert to string for Python interop
    pub fn to_string(&self) -> String {
        match self {
            Device::Cpu => "cpu".to_string(),
            Device::Cuda(id) => format!("cuda:{}", id),
            Device::Metal(_) => "metal".to_string(),
            Device::Vulkan(_) => "vulkan".to_string(),
            Device::OpenCL(id) => format!("opencl:{}", id),
        }
    }

    /// Get optimal memory alignment for this device
    pub const fn memory_alignment(&self) -> usize {
        match self {
            Device::Cpu => 64, // Cache line alignment
            Device::Cuda(_) => 256, // GPU memory alignment
            Device::Metal(_) => 256,
            Device::Vulkan(_) => 256,
            Device::OpenCL(_) => 256,
        }
    }

    /// Check if device supports SIMD operations
    pub const fn supports_simd(&self) -> bool {
        matches!(self, Device::Cpu)
    }

    /// Get device compute capability (for CUDA)
    pub fn compute_capability(&self) -> Option<(u32, u32)> {
        match self {
            Device::Cuda(_) => Some((8, 6)), // Default to RTX 30 series
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_features() {
        assert!(Device::Cuda(0).is_gpu());
        assert!(Device::Cuda(0).supports_async());
        assert!(!Device::Cpu.supports_async());
        assert_eq!(Device::Cuda(0).to_string(), "cuda:0");
    }

    #[tokio::test]
    async fn test_device_memory_capacity() {
        let capacity = Device::Cuda(0).memory_capacity().await.unwrap();
        assert!(capacity > 0);
    }

    #[test]
    fn test_device_alignment() {
        assert_eq!(Device::Cpu.memory_alignment(), 64);
        assert_eq!(Device::Cuda(0).memory_alignment(), 256);
    }
} 