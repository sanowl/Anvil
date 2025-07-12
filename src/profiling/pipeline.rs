use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct PipelineStage {
    pub name: String,
    pub duration: Duration,
    pub input_size: usize,
    pub output_size: usize,
}

#[derive(Debug)]
pub struct PipelineProfiler {
    stages: HashMap<String, Vec<PipelineStage>>,
}

impl PipelineProfiler {
    pub fn new() -> Self {
        Self {
            stages: HashMap::new(),
        }
    }
    
    pub fn record_stage(&mut self, name: String, duration: Duration, input_size: usize, output_size: usize) {
        let stage = PipelineStage {
            name: name.clone(),
            duration,
            input_size,
            output_size,
        };
        
        self.stages.entry(name).or_insert_with(Vec::new).push(stage);
    }
} 