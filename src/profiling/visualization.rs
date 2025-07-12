use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ChartData {
    pub labels: Vec<String>,
    pub values: Vec<f64>,
    pub title: String,
}

#[derive(Debug)]
pub struct ProfilerVisualizer {
    charts: HashMap<String, ChartData>,
}

impl ProfilerVisualizer {
    pub fn new() -> Self {
        Self {
            charts: HashMap::new(),
        }
    }
    
    pub fn add_chart(&mut self, name: String, chart: ChartData) {
        self.charts.insert(name, chart);
    }
    
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str("# Performance Report\n\n");
        
        for (name, chart) in &self.charts {
            report.push_str(&format!("## {}\n", chart.title));
            report.push_str(&format!("Chart: {}\n\n", name));
        }
        
        report
    }
} 