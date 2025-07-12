//! Main trainer implementation for the Anvil ML framework

use std::time::{Duration, Instant};
use std::collections::HashMap;
use crate::{
    autograd::{Variable, AutogradEngine},
    error::{AnvilError, AnvilResult},
    tensor::AdvancedTensor,
};
use super::{
    optimizers::{Optimizer, OptimizerConfig},
    schedulers::{LRScheduler, SchedulerConfig},
    metrics::{TrainingMetrics, MetricTracker},
    checkpoints::{CheckpointManager, Checkpoint},
    mixed_precision::{MixedPrecisionConfig, GradientScaler},
    TrainingCallback, TrainingResult, ValidationResult,
    GradientClipper, GradientAccumulator,
};

/// Training configuration
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub learning_rate: f32,
    pub optimizer_config: OptimizerConfig,
    pub scheduler_config: Option<SchedulerConfig>,
    pub grad_clip_config: Option<(super::ClipType, f32)>,
    pub grad_accumulation_steps: usize,
    pub mixed_precision: Option<MixedPrecisionConfig>,
    pub validation_frequency: usize, // Validate every N epochs
    pub checkpoint_frequency: usize, // Save checkpoint every N epochs
    pub max_checkpoints: usize,
    pub early_stopping_patience: Option<usize>,
    pub weight_decay: f32,
    pub warmup_steps: usize,
    pub log_frequency: usize, // Log every N steps
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            epochs: 100,
            batch_size: 32,
            learning_rate: 1e-3,
            optimizer_config: OptimizerConfig::default(),
            scheduler_config: None,
            grad_clip_config: None,
            grad_accumulation_steps: 1,
            mixed_precision: None,
            validation_frequency: 1,
            checkpoint_frequency: 10,
            max_checkpoints: 5,
            early_stopping_patience: None,
            weight_decay: 0.0,
            warmup_steps: 0,
            log_frequency: 100,
        }
    }
}

/// Training state
#[derive(Debug, Clone)]
pub struct TrainingState {
    pub epoch: usize,
    pub step: usize,
    pub best_loss: f32,
    pub learning_rate: f32,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
    pub metrics: TrainingMetrics,
    pub start_time: Instant,
    pub total_examples: usize,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self {
            epoch: 0,
            step: 0,
            best_loss: f32::INFINITY,
            learning_rate: 1e-3,
            train_loss: 0.0,
            val_loss: None,
            metrics: TrainingMetrics::new(),
            start_time: Instant::now(),
            total_examples: 0,
        }
    }
}

/// Main trainer struct
pub struct Trainer {
    config: TrainingConfig,
    state: TrainingState,
    optimizer: Box<dyn Optimizer>,
    scheduler: Option<Box<dyn LRScheduler>>,
    gradient_clipper: Option<GradientClipper>,
    gradient_accumulator: Option<GradientAccumulator>,
    gradient_scaler: Option<GradientScaler>,
    checkpoint_manager: CheckpointManager,
    metric_tracker: MetricTracker,
    callbacks: Vec<Box<dyn TrainingCallback>>,
    autograd_engine: AutogradEngine,
}

impl Default for Trainer {
    fn default() -> Self {
        let config = TrainingConfig::default();
        let optimizer = config.optimizer_config.create_optimizer();
        
        Self {
            config: config.clone(),
            state: TrainingState::default(),
            optimizer,
            scheduler: None,
            gradient_clipper: None,
            gradient_accumulator: None,
            gradient_scaler: None,
            checkpoint_manager: CheckpointManager::new("./checkpoints"),
            metric_tracker: MetricTracker::new(),
            callbacks: Vec::new(),
            autograd_engine: AutogradEngine::new(),
        }
    }
}

impl Trainer {
    /// Create a new trainer with configuration
    pub fn new(config: TrainingConfig) -> AnvilResult<Self> {
        let optimizer = config.optimizer_config.create_optimizer();
        let scheduler = config.scheduler_config.as_ref().map(|sc| sc.create_scheduler());
        
        let gradient_clipper = config.grad_clip_config.map(|(clip_type, clip_value)| {
            GradientClipper::new(clip_type, clip_value)
        });
        
        let gradient_accumulator = if config.grad_accumulation_steps > 1 {
            Some(GradientAccumulator::new(config.grad_accumulation_steps))
        } else {
            None
        };
        
        let gradient_scaler = config.mixed_precision.as_ref().map(|mp_config| {
            GradientScaler::new(mp_config.clone())
        });
        
        let mut state = TrainingState::default();
        state.learning_rate = config.learning_rate;
        
        Ok(Self {
            config,
            state,
            optimizer,
            scheduler,
            gradient_clipper,
            gradient_accumulator,
            gradient_scaler,
            checkpoint_manager: CheckpointManager::new("./checkpoints"),
            metric_tracker: MetricTracker::new(),
            callbacks: Vec::new(),
            autograd_engine: AutogradEngine::new(),
        })
    }
    
    /// Add a training callback
    pub fn add_callback(&mut self, callback: Box<dyn TrainingCallback>) {
        self.callbacks.push(callback);
    }
    
    /// Train a model
    pub async fn train<Model, Dataset, LossFn>(
        &mut self,
        model: &mut Model,
        train_dataset: &Dataset,
        val_dataset: Option<&Dataset>,
        loss_fn: LossFn,
    ) -> AnvilResult<Vec<TrainingResult>>
    where
        Model: Clone + Send + Sync,
        Dataset: Send + Sync,
        LossFn: Fn(&Variable<f32, 2>, &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>> + Send + Sync,
    {
        let mut results = Vec::new();
        self.state.start_time = Instant::now();
        
        // Call training start callbacks
        for callback in &mut self.callbacks {
            callback.on_train_start(self)?;
        }
        
        for epoch in 0..self.config.epochs {
            self.state.epoch = epoch;
            
            // Call epoch start callbacks
            for callback in &mut self.callbacks {
                callback.on_epoch_start(epoch, self)?;
            }
            
            let epoch_start = Instant::now();
            
            // Training loop
            let train_result = self.train_epoch(model, train_dataset, &loss_fn).await?;
            
            // Validation loop
            let val_result = if epoch % self.config.validation_frequency == 0 {
                if let Some(val_data) = val_dataset {
                    Some(self.validate(model, val_data, &loss_fn).await?)
                } else {
                    None
                }
            } else {
                None
            };
            
            // Update state
            self.state.train_loss = train_result.train_loss;
            self.state.val_loss = val_result.as_ref().map(|vr| vr.loss);
            
            // Learning rate scheduling
            if let Some(scheduler) = &mut self.scheduler {
                let metric = val_result.as_ref().map(|vr| vr.loss).unwrap_or(train_result.train_loss);
                scheduler.step(metric);
                self.state.learning_rate = scheduler.get_lr();
            }
            
            // Create epoch result
            let epoch_result = TrainingResult {
                epoch,
                step: self.state.step,
                train_loss: train_result.train_loss,
                val_loss: val_result.as_ref().map(|vr| vr.loss),
                metrics: train_result.metrics.clone(),
                duration: epoch_start.elapsed(),
            };
            
            results.push(epoch_result.clone());
            
            // Update best loss and save checkpoint if improved
            let current_loss = val_result.as_ref().map(|vr| vr.loss).unwrap_or(train_result.train_loss);
            if current_loss < self.state.best_loss {
                self.state.best_loss = current_loss;
                
                // Save best model checkpoint
                let checkpoint = Checkpoint {
                    epoch,
                    step: self.state.step,
                    loss: current_loss,
                    learning_rate: self.state.learning_rate,
                    optimizer_state: HashMap::new(), // Simplified
                    model_state: HashMap::new(),     // Simplified
                    metrics: train_result.metrics.clone(),
                };
                
                self.checkpoint_manager.save_checkpoint(&checkpoint, "best_model").await?;
                
                // Call improvement callbacks
                for callback in &mut self.callbacks {
                    callback.on_improvement(self.state.best_loss, current_loss, self)?;
                }
            }
            
            // Regular checkpointing
            if epoch % self.config.checkpoint_frequency == 0 {
                let checkpoint = Checkpoint {
                    epoch,
                    step: self.state.step,
                    loss: current_loss,
                    learning_rate: self.state.learning_rate,
                    optimizer_state: HashMap::new(), // Simplified
                    model_state: HashMap::new(),     // Simplified
                    metrics: train_result.metrics.clone(),
                };
                
                self.checkpoint_manager.save_checkpoint(&checkpoint, &format!("epoch_{}", epoch)).await?;
                self.checkpoint_manager.cleanup_old_checkpoints(self.config.max_checkpoints).await?;
            }
            
            // Call epoch end callbacks
            for callback in &mut self.callbacks {
                callback.on_epoch_end(epoch, &epoch_result, self)?;
            }
            
            // Check early stopping
            if let Some(early_stopping) = self.callbacks.iter().any(|cb| {
                // This is simplified - would need proper type checking
                false // placeholder
            }) {
                if early_stopping {
                    println!("Early stopping triggered at epoch {}", epoch);
                    break;
                }
            }
            
            // Logging
            if epoch % 10 == 0 || epoch == self.config.epochs - 1 {
                self.log_epoch_results(&epoch_result, val_result.as_ref());
            }
        }
        
        // Call training end callbacks
        for callback in &mut self.callbacks {
            callback.on_train_end(self)?;
        }
        
        Ok(results)
    }
    
    /// Train for one epoch
    async fn train_epoch<Model, Dataset, LossFn>(
        &mut self,
        model: &mut Model,
        dataset: &Dataset,
        loss_fn: &LossFn,
    ) -> AnvilResult<TrainingResult>
    where
        Model: Clone + Send + Sync,
        Dataset: Send + Sync,
        LossFn: Fn(&Variable<f32, 2>, &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>> + Send + Sync,
    {
        let epoch_start = Instant::now();
        let mut epoch_loss = 0.0;
        let mut num_batches = 0;
        let mut epoch_metrics = TrainingMetrics::new();
        
        // Reset gradient accumulation
        if let Some(grad_acc) = &mut self.gradient_accumulator {
            grad_acc.reset();
        }
        
        // Simulate batches (in real implementation, this would iterate over actual data)
        let num_batches_total = 1000; // Placeholder
        
        for batch_idx in 0..num_batches_total {
            // Call step start callbacks
            for callback in &mut self.callbacks {
                callback.on_step_start(self.state.step, self)?;
            }
            
            // Forward pass (simplified - would use actual model and data)
            let predictions = Variable::zeros(
                crate::tensor::Shape::new([self.config.batch_size, 10]),
                crate::tensor::DType::F32,
                crate::tensor::Device::Cpu,
                true,
            )?;
            
            let targets = Variable::zeros(
                crate::tensor::Shape::new([self.config.batch_size, 10]),
                crate::tensor::DType::F32,
                crate::tensor::Device::Cpu,
                false,
            )?;
            
            // Compute loss
            let loss = loss_fn(&predictions, &targets)?;
            let loss_value = 0.5; // Placeholder loss value
            
            // Backward pass
            self.autograd_engine.backward(&loss)?;
            
            // Get gradients (simplified)
            let gradients = vec![predictions]; // Placeholder gradients
            
            // Gradient accumulation
            if let Some(grad_acc) = &mut self.gradient_accumulator {
                grad_acc.accumulate_gradients(&gradients)?;
                
                if grad_acc.should_step() {
                    let accumulated_grads = grad_acc.get_accumulated_gradients();
                    self.optimizer_step(accumulated_grads)?;
                }
            } else {
                self.optimizer_step(gradients)?;
            }
            
            // Update metrics
            epoch_loss += loss_value;
            num_batches += 1;
            self.state.step += 1;
            self.state.total_examples += self.config.batch_size;
            
            // Call step end callbacks
            for callback in &mut self.callbacks {
                callback.on_step_end(self.state.step, loss_value, self)?;
            }
            
            // Logging
            if self.state.step % self.config.log_frequency == 0 {
                self.log_step_results(batch_idx, loss_value);
            }
        }
        
        Ok(TrainingResult {
            epoch: self.state.epoch,
            step: self.state.step,
            train_loss: epoch_loss / num_batches as f32,
            val_loss: None,
            metrics: epoch_metrics,
            duration: epoch_start.elapsed(),
        })
    }
    
    /// Validate the model
    async fn validate<Model, Dataset, LossFn>(
        &mut self,
        model: &Model,
        dataset: &Dataset,
        loss_fn: &LossFn,
    ) -> AnvilResult<ValidationResult>
    where
        Model: Clone + Send + Sync,
        Dataset: Send + Sync,
        LossFn: Fn(&Variable<f32, 2>, &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>> + Send + Sync,
    {
        let val_start = Instant::now();
        let mut val_loss = 0.0;
        let mut num_batches = 0;
        let mut val_metrics = TrainingMetrics::new();
        
        // Simulate validation batches
        let num_val_batches = 100; // Placeholder
        
        for _batch_idx in 0..num_val_batches {
            // Forward pass (no gradients needed for validation)
            let predictions = Variable::zeros(
                crate::tensor::Shape::new([self.config.batch_size, 10]),
                crate::tensor::DType::F32,
                crate::tensor::Device::Cpu,
                false,
            )?;
            
            let targets = Variable::zeros(
                crate::tensor::Shape::new([self.config.batch_size, 10]),
                crate::tensor::DType::F32,
                crate::tensor::Device::Cpu,
                false,
            )?;
            
            // Compute loss
            let loss = loss_fn(&predictions, &targets)?;
            let loss_value = 0.3; // Placeholder loss value
            
            val_loss += loss_value;
            num_batches += 1;
        }
        
        let result = ValidationResult {
            loss: val_loss / num_batches as f32,
            metrics: val_metrics,
            duration: val_start.elapsed(),
        };
        
        // Call validation callbacks
        for callback in &mut self.callbacks {
            callback.on_validation(&result, self)?;
        }
        
        Ok(result)
    }
    
    /// Perform optimizer step with gradient clipping and scaling
    fn optimizer_step(&mut self, gradients: Vec<Variable<f32, 2>>) -> AnvilResult<()> {
        let mut processed_gradients = gradients;
        
        // Gradient scaling (for mixed precision)
        if let Some(scaler) = &mut self.gradient_scaler {
            processed_gradients = scaler.unscale_gradients(processed_gradients)?;
        }
        
        // Gradient clipping
        if let Some(clipper) = &self.gradient_clipper {
            let grad_norm = clipper.clip_gradients(&mut processed_gradients)?;
            self.metric_tracker.record_scalar("grad_norm", grad_norm);
        }
        
        // Optimizer step
        self.optimizer.step(&processed_gradients, self.state.learning_rate)?;
        
        // Zero gradients
        self.autograd_engine.zero_grad();
        
        Ok(())
    }
    
    /// Log step results
    fn log_step_results(&self, batch_idx: usize, loss: f32) {
        let elapsed = self.state.start_time.elapsed();
        let examples_per_sec = self.state.total_examples as f32 / elapsed.as_secs_f32();
        
        println!(
            "Epoch: {}, Step: {}, Batch: {}, Loss: {:.6}, LR: {:.6}, Examples/sec: {:.2}",
            self.state.epoch,
            self.state.step,
            batch_idx,
            loss,
            self.state.learning_rate,
            examples_per_sec
        );
    }
    
    /// Log epoch results
    fn log_epoch_results(&self, train_result: &TrainingResult, val_result: Option<&ValidationResult>) {
        let elapsed = self.state.start_time.elapsed();
        
        let mut log_msg = format!(
            "Epoch: {}, Train Loss: {:.6}, Duration: {:.2}s",
            train_result.epoch,
            train_result.train_loss,
            train_result.duration.as_secs_f32()
        );
        
        if let Some(val_res) = val_result {
            log_msg.push_str(&format!(", Val Loss: {:.6}", val_res.loss));
        }
        
        log_msg.push_str(&format!(
            ", Total Time: {:.2}s, LR: {:.6}",
            elapsed.as_secs_f32(),
            self.state.learning_rate
        ));
        
        println!("{}", log_msg);
    }
    
    /// Get current training state
    pub fn state(&self) -> &TrainingState {
        &self.state
    }
    
    /// Get training configuration
    pub fn config(&self) -> &TrainingConfig {
        &self.config
    }
    
    /// Get metric tracker
    pub fn metrics(&self) -> &MetricTracker {
        &self.metric_tracker
    }
    
    /// Resume training from checkpoint
    pub async fn resume_from_checkpoint(&mut self, checkpoint_path: &str) -> AnvilResult<()> {
        let checkpoint = self.checkpoint_manager.load_checkpoint(checkpoint_path).await?;
        
        self.state.epoch = checkpoint.epoch;
        self.state.step = checkpoint.step;
        self.state.best_loss = checkpoint.loss;
        self.state.learning_rate = checkpoint.learning_rate;
        self.state.metrics = checkpoint.metrics;
        
        // Restore optimizer state (simplified)
        // In practice, this would restore the actual optimizer state
        
        println!("Resumed training from epoch {} with loss {:.6}", checkpoint.epoch, checkpoint.loss);
        
        Ok(())
    }
    
    /// Save current training state
    pub async fn save_checkpoint(&self, name: &str) -> AnvilResult<()> {
        let checkpoint = Checkpoint {
            epoch: self.state.epoch,
            step: self.state.step,
            loss: self.state.best_loss,
            learning_rate: self.state.learning_rate,
            optimizer_state: HashMap::new(), // Simplified
            model_state: HashMap::new(),     // Simplified
            metrics: self.state.metrics.clone(),
        };
        
        self.checkpoint_manager.save_checkpoint(&checkpoint, name).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_trainer_creation() {
        let config = TrainingConfig {
            epochs: 10,
            batch_size: 16,
            learning_rate: 1e-4,
            ..Default::default()
        };
        
        let trainer = Trainer::new(config).unwrap();
        assert_eq!(trainer.config.epochs, 10);
        assert_eq!(trainer.config.batch_size, 16);
        assert_eq!(trainer.state.learning_rate, 1e-4);
    }
    
    #[test]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.epochs, 100);
        assert_eq!(config.batch_size, 32);
        assert_eq!(config.learning_rate, 1e-3);
    }
}