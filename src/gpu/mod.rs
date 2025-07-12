//! Advanced GPU support with multiple backends

pub mod cuda;
pub mod metal;
pub mod vulkan;
pub mod opencl;
pub mod unified;

pub use unified::*;

/// Check if any GPU backend is available
pub async fn is_available() -> bool {
    cuda::cuda_available() || metal::metal_available() || vulkan::vulkan_available() || opencl::opencl_available()
} 