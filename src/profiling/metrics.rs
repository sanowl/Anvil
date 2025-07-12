//! Performance metrics collection

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use crate::{
    tensor::{AdvancedTensor as Tensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
};

/// Performance metrics collector
pub struct MetricsCollector {
    metrics: Arc<RwLock<HashMap<String, MetricValue>>>,
    start_time: Instant,
}

#[derive(Debug, Clone)]
pub enum MetricValue {
    Counter(u64),
    Gauge(f64),
    Histogram(Vec<f64>),
    Timer(Duration),
}

#[derive(Debug, Clone)]
pub struct ProfilingEvent {
    pub operation: String,
    pub duration: Duration,
    pub timestamp: Instant,
    pub memory_usage: usize,
    pub cpu_usage: f64,
    pub gpu_usage: f64,
}

impl ProfilingEvent {
    pub fn is_bottleneck(&self, threshold: Duration) -> bool {
        self.duration > threshold
    }
    
    pub fn performance_score(&self) -> f64 {
        // Calculate performance score based on multiple metrics
        let time_score = 1.0 / self.duration.as_secs_f64();
        let memory_score = 1.0 / (self.memory_usage as f64 + 1.0);
        let resource_score = (self.cpu_usage + self.gpu_usage) / 2.0;
        
        time_score * 0.5 + memory_score * 0.3 + resource_score * 0.2
    }
}

impl MetricsCollector {
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        }
    }
    
    pub async fn increment_counter(&self, name: &str, value: u64) -> AnvilResult<()> {
        let mut metrics = self.metrics.write().await;
        let current = metrics.get(name).cloned().unwrap_or(MetricValue::Counter(0));
        
        match current {
            MetricValue::Counter(count) => {
                metrics.insert(name.to_string(), MetricValue::Counter(count + value));
            }
            _ => {
                return Err(AnvilError::operation_error("metric", "Metric is not a counter"));
            }
        }
        
        Ok(())
    }
    
    pub async fn set_gauge(&self, name: &str, value: f64) -> AnvilResult<()> {
        let mut metrics = self.metrics.write().await;
        metrics.insert(name.to_string(), MetricValue::Gauge(value));
        Ok(())
    }
    
    pub async fn record_histogram(&self, name: &str, value: f64) -> AnvilResult<()> {
        let mut metrics = self.metrics.write().await;
        let current = metrics.get(name).cloned().unwrap_or(MetricValue::Histogram(Vec::new()));
        
        match current {
            MetricValue::Histogram(mut values) => {
                values.push(value);
                metrics.insert(name.to_string(), MetricValue::Histogram(values));
            }
            _ => {
                return Err(AnvilError::operation_error("metric", "Metric is not a histogram"));
            }
        }
        
        Ok(())
    }
    
    pub async fn record_timer(&self, name: &str, duration: Duration) -> AnvilResult<()> {
        let mut metrics = self.metrics.write().await;
        metrics.insert(name.to_string(), MetricValue::Timer(duration));
        Ok(())
    }
    
    pub async fn get_metrics(&self) -> AnvilResult<HashMap<String, MetricValue>> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }
    
    pub async fn get_summary(&self) -> AnvilResult<MetricsSummary> {
        let metrics = self.metrics.read().await;
        let mut summary = MetricsSummary::new();
        
        for (name, value) in metrics.iter() {
            match value {
                MetricValue::Counter(count) => {
                    summary.counters.insert(name.clone(), *count);
                }
                MetricValue::Gauge(value) => {
                    summary.gauges.insert(name.clone(), *value);
                }
                MetricValue::Histogram(values) => {
                    if !values.is_empty() {
                        summary.histograms.insert(name.clone(), HistogramStats {
                            count: values.len(),
                            min: values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
                            max: values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
                            mean: values.iter().sum::<f64>() / values.len() as f64,
                            p95: self.percentile(values, 0.95),
                            p99: self.percentile(values, 0.99),
                        });
                    }
                }
                MetricValue::Timer(duration) => {
                    summary.timers.insert(name.clone(), duration.as_millis() as f64);
                }
            }
        }
        
        Ok(summary)
    }
    
    fn percentile(&self, values: &[f64], p: f64) -> f64 {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = (sorted.len() as f64 * p) as usize;
        sorted[index.min(sorted.len() - 1)]
    }
}

#[derive(Debug)]
pub struct MetricsSummary {
    pub counters: HashMap<String, u64>,
    pub gauges: HashMap<String, f64>,
    pub histograms: HashMap<String, HistogramStats>,
    pub timers: HashMap<String, f64>,
}

impl MetricsSummary {
    pub fn new() -> Self {
        Self {
            counters: HashMap::new(),
            gauges: HashMap::new(),
            histograms: HashMap::new(),
            timers: HashMap::new(),
        }
    }
}

#[derive(Debug)]
pub struct HistogramStats {
    pub count: usize,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Training metrics
pub struct TrainingMetrics {
    collector: MetricsCollector,
}

impl TrainingMetrics {
    pub fn new() -> Self {
        Self {
            collector: MetricsCollector::new(),
        }
    }
    
    pub async fn record_epoch(&self, epoch: usize, loss: f64, accuracy: f64) -> AnvilResult<()> {
        self.collector.set_gauge(&format!("epoch_{}_loss", epoch), loss).await?;
        self.collector.set_gauge(&format!("epoch_{}_accuracy", epoch), accuracy).await?;
        Ok(())
    }
    
    pub async fn record_batch(&self, batch_size: usize, batch_time: Duration) -> AnvilResult<()> {
        self.collector.increment_counter("total_batches", 1).await?;
        self.collector.record_timer("batch_time", batch_time).await?;
        self.collector.set_gauge("current_batch_size", batch_size as f64).await?;
        Ok(())
    }
    
    pub async fn record_memory_usage(&self, cpu_memory: usize, gpu_memory: usize) -> AnvilResult<()> {
        self.collector.set_gauge("cpu_memory_mb", cpu_memory as f64 / 1024.0 / 1024.0).await?;
        self.collector.set_gauge("gpu_memory_mb", gpu_memory as f64 / 1024.0 / 1024.0).await?;
        Ok(())
    }
    
    pub async fn get_summary(&self) -> AnvilResult<MetricsSummary> {
        self.collector.get_summary().await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_metrics_collector() {
        let collector = MetricsCollector::new();
        
        collector.increment_counter("test_counter", 5).await.unwrap();
        collector.set_gauge("test_gauge", 42.5).await.unwrap();
        collector.record_histogram("test_hist", 1.0).await.unwrap();
        collector.record_timer("test_timer", Duration::from_millis(100)).await.unwrap();
        
        let summary = collector.get_summary().await.unwrap();
        assert_eq!(summary.counters.get("test_counter"), Some(&5));
        assert_eq!(summary.gauges.get("test_gauge"), Some(&42.5));
    }
    
    #[tokio::test]
    async fn test_training_metrics() {
        let metrics = TrainingMetrics::new();
        
        metrics.record_epoch(1, 0.5, 0.85).await.unwrap();
        metrics.record_batch(32, Duration::from_millis(50)).await.unwrap();
        metrics.record_memory_usage(1024 * 1024 * 100, 1024 * 1024 * 200).await.unwrap();
        
        let summary = metrics.get_summary().await.unwrap();
        assert!(summary.counters.contains_key("total_batches"));
        assert!(summary.gauges.contains_key("epoch_1_loss"));
    }
} 