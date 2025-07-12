use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct TraceEvent {
    pub name: String,
    pub timestamp: Instant,
    pub data: HashMap<String, String>,
}

#[derive(Debug)]
pub struct Tracer {
    events: Vec<TraceEvent>,
    enabled: bool,
}

impl Tracer {
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            enabled: true,
        }
    }
    
    pub fn trace(&mut self, name: String, data: HashMap<String, String>) {
        if self.enabled {
            let event = TraceEvent {
                name,
                timestamp: Instant::now(),
                data,
            };
            self.events.push(event);
        }
    }
    
    pub fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
    }
} 