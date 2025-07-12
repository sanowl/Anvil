use std::sync::Arc;
use parking_lot::Mutex;
use crate::error::{AnvilError, AnvilResult};

pub trait MemoryPool: Send + Sync {
    fn allocate(&self, size: usize) -> MemoryGuard;
    fn deallocate(&self, size: usize);
}

pub struct MemoryGuard {
    size: usize,
    pool: Arc<dyn MemoryPool>,
}

impl MemoryGuard {
    pub fn new(size: usize, pool: Arc<dyn MemoryPool>) -> Self {
        Self { size, pool }
    }
}

impl Drop for MemoryGuard {
    fn drop(&mut self) {
        self.pool.deallocate(self.size);
    }
}

pub struct DefaultMemoryPool {
    allocated: Arc<Mutex<usize>>,
}

impl DefaultMemoryPool {
    pub fn new() -> Self {
        Self {
            allocated: Arc::new(Mutex::new(0)),
        }
    }
}

impl MemoryPool for DefaultMemoryPool {
    fn allocate(&self, size: usize) -> MemoryGuard {
        *self.allocated.lock() += size;
        MemoryGuard::new(size, Arc::new(self.clone()))
    }
    
    fn deallocate(&self, size: usize) {
        *self.allocated.lock() -= size;
    }
}

impl Clone for DefaultMemoryPool {
    fn clone(&self) -> Self {
        Self {
            allocated: self.allocated.clone(),
        }
    }
}

pub fn get_memory_manager() -> Arc<dyn MemoryPool> {
    Arc::new(DefaultMemoryPool::new())
} 