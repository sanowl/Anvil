//! Training metrics and tracking

use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Training metrics collection
#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    scalars: HashMap<String, f32>,
    timings: HashMap<String, Duration>,
    counters: HashMap<String, u64>,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            scalars: HashMap::new(),
            timings: HashMap::new(),
            counters: HashMap::new(),
        }
    }
    
    pub fn record_scalar(&mut self, name: &str, value: f32) {
        self.scalars.insert(name.to_string(), value);
    }
    
    pub fn record_timing(&mut self, name: &str, duration: Duration) {
        self.timings.insert(name.to_string(), duration);
    }
    
    pub fn increment_counter(&mut self, name: &str) {
        let count = self.counters.get(name).unwrap_or(&0) + 1;
        self.counters.insert(name.to_string(), count);
    }
    
    pub fn get_scalar(&self, name: &str) -> Option<f32> {
        self.scalars.get(name).copied()
    }
    
    pub fn get_timing(&self, name: &str) -> Option<Duration> {
        self.timings.get(name).copied()
    }
    
    pub fn get_counter(&self, name: &str) -> Option<u64> {
        self.counters.get(name).copied()
    }
    
    pub fn all_scalars(&self) -> &HashMap<String, f32> {
        &self.scalars
    }
    
    pub fn all_timings(&self) -> &HashMap<String, Duration> {
        &self.timings
    }
    
    pub fn all_counters(&self) -> &HashMap<String, u64> {
        &self.counters
    }
}

/// Metric tracker for training
pub struct MetricTracker {
    current_metrics: TrainingMetrics,
    history: Vec<TrainingMetrics>,
    start_time: Instant,
}

impl MetricTracker {
    pub fn new() -> Self {
        Self {
            current_metrics: TrainingMetrics::new(),
            history: Vec::new(),
            start_time: Instant::now(),
        }
    }
    
    pub fn record_scalar(&mut self, name: &str, value: f32) {
        self.current_metrics.record_scalar(name, value);
    }
    
    pub fn record_timing(&mut self, name: &str, duration: Duration) {
        self.current_metrics.record_timing(name, duration);
    }
    
    pub fn increment_counter(&mut self, name: &str) {
        self.current_metrics.increment_counter(name);
    }
    
    pub fn step(&mut self) {
        self.history.push(self.current_metrics.clone());
        self.current_metrics = TrainingMetrics::new();
    }
    
    pub fn current(&self) -> &TrainingMetrics {
        &self.current_metrics
    }
    
    pub fn history(&self) -> &[TrainingMetrics] {
        &self.history
    }
    
    pub fn total_time(&self) -> Duration {
        self.start_time.elapsed()
    }
}