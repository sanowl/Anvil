use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use std::thread;

#[derive(Debug, Clone)]
pub struct SyncPoint {
    pub name: String,
    pub timestamp: Instant,
    pub thread_id: u64,
}

#[derive(Debug)]
pub struct SynchronizationProfiler {
    sync_points: Arc<Mutex<HashMap<String, Vec<SyncPoint>>>>,
}

impl SynchronizationProfiler {
    pub fn new() -> Self {
        Self {
            sync_points: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    pub fn record_sync_point(&self, name: String) {
        let thread_id = thread::current().id();
        let thread_id_numeric = unsafe { 
            std::mem::transmute::<std::thread::ThreadId, u64>(thread_id) 
        };
        
        let sync_point = SyncPoint {
            name: name.clone(),
            timestamp: Instant::now(),
            thread_id: thread_id_numeric,
        };
        
        if let Ok(mut points) = self.sync_points.lock() {
            points.entry(name).or_insert_with(Vec::new).push(sync_point);
        }
    }
} 