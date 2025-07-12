//! Safety utilities for memory and thread safety

use std::sync::{Arc, RwLock};
use std::collections::HashMap;
use crate::error::{AnvilError, AnvilResult};

/// Memory safety checker for tensor operations
pub struct MemorySafetyChecker {
    allocations: RwLock<HashMap<usize, AllocationInfo>>,
}

#[derive(Debug, Clone)]
struct AllocationInfo {
    size: usize,
    device: String,
    allocated_at: std::time::Instant,
}

impl MemorySafetyChecker {
    pub fn new() -> Self {
        Self {
            allocations: RwLock::new(HashMap::new()),
        }
    }

    /// Check if a memory allocation is safe
    pub fn check_allocation(&self, size: usize, device: &str) -> AnvilResult<()> {
        if size == 0 {
            return Err(AnvilError::InvalidInput("Cannot allocate zero bytes".to_string()));
        }

        // Check for reasonable allocation size (1GB limit)
        if size > 1024 * 1024 * 1024 {
            return Err(AnvilError::InvalidInput("Allocation size too large".to_string()));
        }

        let mut allocations = self.allocations.write().map_err(|e| AnvilError::InternalError(e.to_string()))?;
        allocations.insert(size, AllocationInfo {
            size,
            device: device.to_string(),
            allocated_at: std::time::Instant::now(),
        });

        Ok(())
    }

    /// Check if a memory deallocation is safe
    pub fn check_deallocation(&self, size: usize) -> AnvilResult<()> {
        let allocations = self.allocations.read().map_err(|e| AnvilError::InternalError(e.to_string()))?;
        if !allocations.contains_key(&size) {
            return Err(AnvilError::InvalidInput("Attempting to deallocate unallocated memory".to_string()));
        }

        Ok(())
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> AnvilResult<MemoryStats> {
        let allocations = self.allocations.read().map_err(|e| AnvilError::InternalError(e.to_string()))?;
        let total_allocated: usize = allocations.values().map(|info| info.size).sum();
        let allocation_count = allocations.len();
        
        Ok(MemoryStats {
            total_allocated,
            allocation_count,
            peak_usage: total_allocated, // Simplified
        })
    }
}

#[derive(Debug, Clone)]
pub struct MemoryStats {
    pub total_allocated: usize,
    pub allocation_count: usize,
    pub peak_usage: usize,
}

/// Thread safety checker for concurrent operations
pub struct ThreadSafetyChecker {
    active_operations: RwLock<HashMap<String, OperationInfo>>,
}

#[derive(Debug, Clone)]
struct OperationInfo {
    operation_type: String,
    started_at: std::time::Instant,
    thread_id: String,
}

impl ThreadSafetyChecker {
    pub fn new() -> Self {
        Self {
            active_operations: RwLock::new(HashMap::new()),
        }
    }

    /// Check if an operation can be safely executed
    pub fn check_operation(&self, operation_id: &str, operation_type: &str) -> AnvilResult<()> {
        let mut operations = self.active_operations.write().map_err(|e| AnvilError::InternalError(e.to_string()))?;
        
        if operations.contains_key(operation_id) {
            return Err(AnvilError::InvalidInput(
                format!("Operation {} already in progress", operation_id)
            ));
        }

        operations.insert(operation_id.to_string(), OperationInfo {
            operation_type: operation_type.to_string(),
            started_at: std::time::Instant::now(),
            thread_id: format!("{:?}", std::thread::current().id()),
        });

        Ok(())
    }

    /// Mark an operation as completed
    pub fn complete_operation(&self, operation_id: &str) -> AnvilResult<()> {
        let mut operations = self.active_operations.write().map_err(|e| AnvilError::InternalError(e.to_string()))?;
        
        if operations.remove(operation_id).is_none() {
            return Err(AnvilError::InvalidInput(
                format!("Operation {} not found", operation_id)
            ));
        }

        Ok(())
    }

    /// Get active operations
    pub fn get_active_operations(&self) -> AnvilResult<Vec<String>> {
        let operations = self.active_operations.read().map_err(|e| AnvilError::InternalError(e.to_string()))?;
        Ok(operations.keys().cloned().collect())
    }
}

/// Global safety checkers
lazy_static::lazy_static! {
    pub static ref MEMORY_SAFETY: Arc<MemorySafetyChecker> = Arc::new(MemorySafetyChecker::new());
    pub static ref THREAD_SAFETY: Arc<ThreadSafetyChecker> = Arc::new(ThreadSafetyChecker::new());
}

/// Check if the current operation is safe to execute
pub fn check_operation_safety(operation_id: &str, operation_type: &str) -> AnvilResult<()> {
    THREAD_SAFETY.check_operation(operation_id, operation_type)
}

/// Mark an operation as completed
pub fn complete_operation(operation_id: &str) -> AnvilResult<()> {
    THREAD_SAFETY.complete_operation(operation_id)
}

/// Get current memory statistics
pub fn get_memory_stats() -> MemoryStats {
    MEMORY_SAFETY.get_memory_stats().unwrap() // Unwrap is safe because it's a lazy_static
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_safety_checker() {
        let checker = MemorySafetyChecker::new();
        
        assert!(checker.check_allocation(1024, "cpu").is_ok());
        assert!(checker.check_allocation(0, "cpu").is_err());
        assert!(checker.check_allocation(2 * 1024 * 1024 * 1024, "cpu").is_err());
    }

    #[test]
    fn test_thread_safety_checker() {
        let checker = ThreadSafetyChecker::new();
        
        assert!(checker.check_operation("op1", "matmul").is_ok());
        assert!(checker.check_operation("op1", "matmul").is_err()); // Duplicate
        assert!(checker.complete_operation("op1").is_ok());
        assert!(checker.complete_operation("op1").is_err()); // Already completed
    }
} 