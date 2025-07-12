use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Debug, Clone)]
pub struct ProfileEvent {
    pub name: String,
    pub start_time: Instant,
    pub end_time: Option<Instant>,
    pub duration: Option<Duration>,
}

#[derive(Debug)]
pub struct Profiler {
    events: HashMap<String, Vec<ProfileEvent>>,
    active_events: HashMap<String, Instant>,
}

impl Profiler {
    pub fn new() -> Self {
        Self {
            events: HashMap::new(),
            active_events: HashMap::new(),
        }
    }
    
    pub fn start_event(&mut self, name: String) {
        self.active_events.insert(name.clone(), Instant::now());
    }
    
    pub fn end_event(&mut self, name: String) {
        if let Some(start_time) = self.active_events.remove(&name) {
            let end_time = Instant::now();
            let duration = end_time.duration_since(start_time);
            
            let event = ProfileEvent {
                name: name.clone(),
                start_time,
                end_time: Some(end_time),
                duration: Some(duration),
            };
            
            self.events.entry(name).or_insert_with(Vec::new).push(event);
        }
    }
} 