//! Advanced memory allocator with SIMD alignment and pooling

use std::{
    collections::HashMap,
    ptr::NonNull,
    alloc::{alloc, dealloc, Layout},
};
use parking_lot::RwLock as PLRwLock;
use crate::error::{AnvilError, AnvilResult};
use super::devices::Device;

/// Advanced memory allocator with SIMD alignment and pooling
pub struct AdvancedAllocator {
    pools: PLRwLock<HashMap<usize, Vec<NonNull<u8>>>>,
    alignment: usize,
    device: Device,
}

impl AdvancedAllocator {
    pub fn new(device: Device, alignment: usize) -> Self {
        Self {
            pools: PLRwLock::new(HashMap::new()),
            alignment,
            device,
        }
    }

    /// Allocate memory with proper alignment
    pub unsafe fn allocate(&self, size: usize) -> AnvilResult<NonNull<u8>> {
        // Try to get from pool first
        {
            let mut pools = self.pools.write();
            if let Some(pool) = pools.get_mut(&size) {
                if let Some(ptr) = pool.pop() {
                    return Ok(ptr);
                }
            }
        }

        // Allocate new memory
        let layout = Layout::from_size_align(size, self.alignment)
            .map_err(|e| AnvilError::MemoryError(format!("Invalid layout: {}", e)))?;
        
        let ptr = alloc(layout);
        if ptr.is_null() {
            return Err(AnvilError::MemoryError("Allocation failed".to_string()));
        }

        Ok(NonNull::new_unchecked(ptr))
    }

    /// Deallocate memory and return to pool
    pub unsafe fn deallocate(&self, size: usize, ptr: NonNull<u8>) {
        // Return to pool for reuse
        let mut pools = self.pools.write();
        pools.entry(size).or_insert_with(Vec::new).push(ptr);
        
        // Limit pool size to prevent memory leaks
        if let Some(pool) = pools.get_mut(&size) {
            if pool.len() > 100 {
                if let Some(old_ptr) = pool.pop() {
                    let layout = Layout::from_size_align_unchecked(size, self.alignment);
                    dealloc(old_ptr.as_ptr(), layout);
                }
            }
        }
    }

    /// Clear all pools (for cleanup)
    pub fn clear_pools(&self) {
        let mut pools = self.pools.write();
        for (size, pool) in pools.iter_mut() {
            for ptr in pool.drain(..) {
                unsafe {
                    let layout = Layout::from_size_align_unchecked(*size, self.alignment);
                    dealloc(ptr.as_ptr(), layout);
                }
            }
        }
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> AllocatorStats {
        let pools = self.pools.read();
        let mut total_allocated = 0;
        let mut total_pooled = 0;
        let mut pool_count = 0;

        for (size, pool) in pools.iter() {
            total_allocated += size * 100; // Estimate based on pool size
            total_pooled += size * pool.len();
            pool_count += pool.len();
        }

        AllocatorStats {
            total_allocated,
            total_pooled,
            pool_count,
            alignment: self.alignment,
        }
    }
}

impl Drop for AdvancedAllocator {
    fn drop(&mut self) {
        self.clear_pools();
    }
}

/// Memory allocation statistics
#[derive(Debug, Clone)]
pub struct AllocatorStats {
    pub total_allocated: usize,
    pub total_pooled: usize,
    pub pool_count: usize,
    pub alignment: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_allocator_creation() {
        let allocator = AdvancedAllocator::new(Device::Cpu, 16);
        assert_eq!(allocator.alignment, 16);
    }

    #[test]
    fn test_memory_allocation() {
        let allocator = AdvancedAllocator::new(Device::Cpu, 16);
        
        unsafe {
            let ptr = allocator.allocate(1024).unwrap();
            assert!(!ptr.as_ptr().is_null());
            
            allocator.deallocate(1024, ptr);
        }
    }

    #[test]
    fn test_memory_stats() {
        let allocator = AdvancedAllocator::new(Device::Cpu, 16);
        let stats = allocator.memory_stats();
        
        assert_eq!(stats.alignment, 16);
        assert_eq!(stats.pool_count, 0);
    }
} 