//! Configuration management for the Anvil framework with hot reloading

use lazy_static::lazy_static;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;
use crate::error::{AnvilError, AnvilResult};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    pub kernel_fusion: bool,
    pub num_worker_threads: usize,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            kernel_fusion: true,
            num_worker_threads: 4,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryConfig {
    pub max_pool_size: usize,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            max_pool_size: 1024 * 1024 * 1024, // 1GB
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    pub enabled: bool,
    pub device_id: u32,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            device_id: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub learning_rate: f32,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 32,
            learning_rate: 0.001,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedConfig {
    pub enabled: bool,
    pub num_nodes: usize,
}

impl Default for DistributedConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            num_nodes: 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProfilingConfig {
    pub enabled: bool,
    pub output_path: String,
}

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            output_path: "profiles/".to_string(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyConfig {
    pub bounds_checking: bool,
    pub overflow_checking: bool,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            bounds_checking: true,
            overflow_checking: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DevelopmentConfig {
    pub debug_mode: bool,
    pub verbose_logging: bool,
}

impl Default for DevelopmentConfig {
    fn default() -> Self {
        Self {
            debug_mode: false,
            verbose_logging: false,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnvilConfig {
    pub performance: PerformanceConfig,
    pub memory: MemoryConfig,
    pub gpu: GpuConfig,
    pub training: TrainingConfig,
    pub distributed: DistributedConfig,
    pub profiling: ProfilingConfig,
    pub safety: SafetyConfig,
    pub development: DevelopmentConfig,
}

impl Default for AnvilConfig {
    fn default() -> Self {
        Self {
            performance: PerformanceConfig::default(),
            memory: MemoryConfig::default(),
            gpu: GpuConfig::default(),
            training: TrainingConfig::default(),
            distributed: DistributedConfig::default(),
            profiling: ProfilingConfig::default(),
            safety: SafetyConfig::default(),
            development: DevelopmentConfig::default(),
        }
    }
}

/// Configuration manager with hot reloading support
pub struct ConfigManager {
    config: Arc<RwLock<AnvilConfig>>,
    config_file: Option<PathBuf>,
    last_modified: Instant,
    watchers: Vec<tokio::sync::watch::Sender<AnvilConfig>>,
    custom_settings: HashMap<String, serde_json::Value>,
}

impl ConfigManager {
    pub fn new() -> Self {
        Self {
            config: Arc::new(RwLock::new(AnvilConfig::default())),
            config_file: None,
            last_modified: Instant::now(),
            watchers: Vec::new(),
            custom_settings: HashMap::new(),
        }
    }
    
    pub fn with_config_file(mut self, path: PathBuf) -> Self {
        self.config_file = Some(path);
        self
    }
    
    pub fn load_from_file(&mut self, path: &PathBuf) -> AnvilResult<()> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| AnvilError::operation_error("config", &format!("Failed to read config file: {}", e)))?;
        
        let config: AnvilConfig = serde_json::from_str(&content)
            .map_err(|e| AnvilError::operation_error("config", &format!("Failed to parse config file: {}", e)))?;
        
        *self.config.write().unwrap() = config;
        self.config_file = Some(path.clone());
        self.last_modified = Instant::now();
        
        tracing::info!("Configuration loaded from {:?}", path);
        self.notify_watchers();
        
        Ok(())
    }
    
    pub fn save_to_file(&self, path: &PathBuf) -> AnvilResult<()> {
        let config = self.config.read().unwrap();
        let content = serde_json::to_string_pretty(&*config)
            .map_err(|e| AnvilError::operation_error("config", &format!("Failed to serialize config: {}", e)))?;
        
        std::fs::write(path, content)
            .map_err(|e| AnvilError::operation_error("config", &format!("Failed to write config file: {}", e)))?;
        
        tracing::info!("Configuration saved to {:?}", path);
        Ok(())
    }
    
    pub fn get_config(&self) -> AnvilConfig {
        self.config.read().unwrap().clone()
    }
    
    pub fn update_config<F>(&mut self, f: F) -> AnvilResult<()>
    where
        F: FnOnce(&mut AnvilConfig),
    {
        {
            let mut config = self.config.write().unwrap();
            f(&mut *config);
        }
        
        self.last_modified = Instant::now();
        self.notify_watchers();
        
        // Save to file if configured
        if let Some(ref path) = self.config_file {
            self.save_to_file(path)?;
        }
        
        Ok(())
    }
    
    pub fn watch_config(&mut self) -> tokio::sync::watch::Receiver<AnvilConfig> {
        let (sender, receiver) = tokio::sync::watch::channel(self.get_config());
        self.watchers.push(sender);
        receiver
    }
    
    pub fn set_custom_setting(&mut self, key: &str, value: serde_json::Value) {
        self.custom_settings.insert(key.to_string(), value);
    }
    
    pub fn get_custom_setting(&self, key: &str) -> Option<&serde_json::Value> {
        self.custom_settings.get(key)
    }
    
    pub fn set_config(&mut self, config: AnvilConfig) {
        *self.config.write().unwrap() = config;
        self.last_modified = Instant::now();
        self.notify_watchers();
        if let Some(ref path) = self.config_file {
            let _ = self.save_to_file(path);
        }
    }
    
    fn notify_watchers(&mut self) {
        let config = self.get_config();
        self.watchers.retain_mut(|sender| {
            sender.send(config.clone()).is_ok()
        });
    }
    
    pub fn check_for_updates(&mut self) -> AnvilResult<bool> {
        if let Some(path) = &self.config_file {
            if let Ok(metadata) = std::fs::metadata(path) {
                if let Ok(_modified) = metadata.modified() {
                    let modified = Instant::now(); // Simplified for this example
                    if modified > self.last_modified {
                        let path_clone = path.clone();
                        self.load_from_file(&path_clone)?;
                        return Ok(true);
                    }
                }
            }
        }
        Ok(false)
    }
}

/// Global configuration instance
lazy_static! {
    static ref CONFIG_MANAGER: Arc<RwLock<ConfigManager>> = Arc::new(RwLock::new(ConfigManager::new()));
}

/// Get the global configuration
pub fn get_config() -> AnvilConfig {
    CONFIG_MANAGER.read().unwrap().get_config()
}

/// Update the global configuration
pub fn update_config<F>(f: F) -> AnvilResult<()>
where
    F: FnOnce(&mut AnvilConfig),
{
    let mut manager = CONFIG_MANAGER.write().unwrap();
    manager.update_config(f)
}

/// Load configuration from file
pub fn load_config_from_file(path: &PathBuf) -> AnvilResult<()> {
    let mut manager = CONFIG_MANAGER.write().unwrap();
    manager.load_from_file(path)?;
    
    // Update global config
    let config = manager.get_config();
    // *ANVIL_CONFIG.write().unwrap() = config; // This line is removed
    
    Ok(())
}

/// Watch for configuration changes
pub fn watch_config() -> tokio::sync::watch::Receiver<AnvilConfig> {
    let mut manager = CONFIG_MANAGER.write().unwrap();
    manager.watch_config()
}

/// Create a configuration builder for easy setup
pub struct ConfigBuilder {
    config: AnvilConfig,
}

impl ConfigBuilder {
    pub fn new() -> Self {
        Self {
            config: AnvilConfig::default(),
        }
    }
    
    pub fn performance(mut self, f: impl FnOnce(&mut PerformanceConfig)) -> Self {
        f(&mut self.config.performance);
        self
    }
    
    pub fn memory(mut self, f: impl FnOnce(&mut MemoryConfig)) -> Self {
        f(&mut self.config.memory);
        self
    }
    
    pub fn gpu(mut self, f: impl FnOnce(&mut GpuConfig)) -> Self {
        f(&mut self.config.gpu);
        self
    }
    
    pub fn training(mut self, f: impl FnOnce(&mut TrainingConfig)) -> Self {
        f(&mut self.config.training);
        self
    }
    
    pub fn distributed(mut self, f: impl FnOnce(&mut DistributedConfig)) -> Self {
        f(&mut self.config.distributed);
        self
    }
    
    pub fn profiling(mut self, f: impl FnOnce(&mut ProfilingConfig)) -> Self {
        f(&mut self.config.profiling);
        self
    }
    
    pub fn safety(mut self, f: impl FnOnce(&mut SafetyConfig)) -> Self {
        f(&mut self.config.safety);
        self
    }
    
    pub fn development(mut self, f: impl FnOnce(&mut DevelopmentConfig)) -> Self {
        f(&mut self.config.development);
        self
    }
    
    pub fn build(self) -> AnvilConfig {
        self.config
    }
    
    pub fn apply(self) -> AnvilResult<()> {
        update_config(|config| *config = self.config)
    }
}



#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_config_builder() {
        let config = ConfigBuilder::new()
            .performance(|p| {
                p.kernel_fusion = false;
                p.num_worker_threads = 4;
            })
            .memory(|m| {
                m.max_pool_size = 512 * 1024 * 1024; // 512MB
            })
            .build();
        
        assert!(!config.performance.kernel_fusion);
        assert_eq!(config.performance.num_worker_threads, 4);
        assert_eq!(config.memory.max_pool_size, 512 * 1024 * 1024);
    }

    #[test]
    fn test_config_manager() {
        let mut manager = ConfigManager::new();
        let config = manager.get_config();
        assert_eq!(config.performance.kernel_fusion, true);
        
        manager.update_config(|c| {
            c.performance.kernel_fusion = false;
        }).unwrap();
        
        let updated_config = manager.get_config();
        assert!(!updated_config.performance.kernel_fusion);
    }

    #[test]
    fn test_config_file_io() {
        let temp_dir = tempdir().unwrap();
        let config_path = temp_dir.path().join("anvil_config.json");
        
        let config = ConfigBuilder::new()
            .performance(|p| p.kernel_fusion = false)
            .build();
        
        // Save config
        let content = serde_json::to_string_pretty(&config).unwrap();
        std::fs::write(&config_path, content).unwrap();
        
        // Load config
        let mut manager = ConfigManager::new();
        manager.load_from_file(&config_path).unwrap();
        
        let loaded_config = manager.get_config();
        assert!(!loaded_config.performance.kernel_fusion);
    }
} 