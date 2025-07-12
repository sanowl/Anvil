//! Advanced Tensor Core with SIMD, Zero-Copy, Async, and Python Interop
//! 
//! This module provides a production-grade tensor implementation with:
//! - SIMD-accelerated operations with auto-vectorization
//! - Zero-copy memory management and memory-mapped tensors
//! - Async device transfers and operations
//! - Custom allocators and memory pools for performance
//! - Compile-time and runtime shape/type/device safety
//! - Python/NumPy interop hooks for PyO3 bindings
//! - Advanced trait-based extensibility

pub mod core;
pub mod storage;
pub mod ops;
pub mod simd;
pub mod devices;

// Re-export main types for convenience
pub use core::{Shape, DType, AdvancedTensor};
pub use storage::AdvancedTensorStorage;
pub use devices::Device;
pub use ops::AdvancedAllocator; 