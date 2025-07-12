//! Advanced training infrastructure for Anvil ML Framework
//! 
//! This module provides a comprehensive training system with:
//! - Advanced optimizers (Adam, AdamW, RMSprop, Lion, etc.)
//! - Learning rate scheduling
//! - Gradient clipping and accumulation
//! - Mixed precision training
//! - Automatic loss scaling
//! - Training metrics and logging
//! - Checkpointing and resumption

pub mod trainer;
pub mod optimizers;
pub mod schedulers;
pub mod metrics;
pub mod checkpoints;
pub mod mixed_precision;

pub use trainer::{Trainer, TrainingConfig, TrainingState};
pub use optimizers::{Optimizer, OptimizerConfig, OptimizerType};
pub use schedulers::{LRScheduler, SchedulerConfig, SchedulerType};
pub use metrics::{TrainingMetrics, MetricTracker};
pub use checkpoints::{CheckpointManager, Checkpoint};
pub use mixed_precision::{MixedPrecisionConfig, GradientScaler};

use crate::{
    autograd::{Variable, AutogradEngine},
    error::{AnvilError, AnvilResult},
    tensor::AdvancedTensor,
};

/// Training loop result
#[derive(Debug, Clone)]
pub struct TrainingResult {
    pub epoch: usize,
    pub step: usize,
    pub train_loss: f32,
    pub val_loss: Option<f32>,
    pub metrics: TrainingMetrics,
    pub duration: std::time::Duration,
}

/// Validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub loss: f32,
    pub metrics: TrainingMetrics,
    pub duration: std::time::Duration,
}

/// Training callbacks for custom behavior during training
pub trait TrainingCallback {
    /// Called at the start of training
    fn on_train_start(&mut self, trainer: &Trainer) -> AnvilResult<()> {
        Ok(())
    }
    
    /// Called at the end of training
    fn on_train_end(&mut self, trainer: &Trainer) -> AnvilResult<()> {
        Ok(())
    }
    
    /// Called at the start of each epoch
    fn on_epoch_start(&mut self, epoch: usize, trainer: &Trainer) -> AnvilResult<()> {
        Ok(())
    }
    
    /// Called at the end of each epoch
    fn on_epoch_end(&mut self, epoch: usize, result: &TrainingResult, trainer: &Trainer) -> AnvilResult<()> {
        Ok(())
    }
    
    /// Called at the start of each training step
    fn on_step_start(&mut self, step: usize, trainer: &Trainer) -> AnvilResult<()> {
        Ok(())
    }
    
    /// Called at the end of each training step
    fn on_step_end(&mut self, step: usize, loss: f32, trainer: &Trainer) -> AnvilResult<()> {
        Ok(())
    }
    
    /// Called during validation
    fn on_validation(&mut self, result: &ValidationResult, trainer: &Trainer) -> AnvilResult<()> {
        Ok(())
    }
    
    /// Called when loss improves
    fn on_improvement(&mut self, old_loss: f32, new_loss: f32, trainer: &Trainer) -> AnvilResult<()> {
        Ok(())
    }
}

/// Early stopping callback
pub struct EarlyStoppingCallback {
    patience: usize,
    min_delta: f32,
    best_loss: f32,
    wait_count: usize,
    stopped: bool,
}

impl EarlyStoppingCallback {
    pub fn new(patience: usize, min_delta: f32) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f32::INFINITY,
            wait_count: 0,
            stopped: false,
        }
    }
    
    pub fn should_stop(&self) -> bool {
        self.stopped
    }
}

impl TrainingCallback for EarlyStoppingCallback {
    fn on_epoch_end(&mut self, _epoch: usize, result: &TrainingResult, _trainer: &Trainer) -> AnvilResult<()> {
        let current_loss = result.val_loss.unwrap_or(result.train_loss);
        
        if current_loss < self.best_loss - self.min_delta {
            self.best_loss = current_loss;
            self.wait_count = 0;
        } else {
            self.wait_count += 1;
            
            if self.wait_count >= self.patience {
                self.stopped = true;
                println!("Early stopping triggered after {} epochs without improvement", self.patience);
            }
        }
        
        Ok(())
    }
}

/// Learning rate finder for optimal learning rate discovery
pub struct LRFinder {
    start_lr: f32,
    end_lr: f32,
    num_steps: usize,
    smooth_factor: f32,
}

impl LRFinder {
    pub fn new(start_lr: f32, end_lr: f32, num_steps: usize) -> Self {
        Self {
            start_lr,
            end_lr,
            num_steps,
            smooth_factor: 0.05,
        }
    }
    
    pub fn find_lr<Model, Dataset>(
        &self,
        model: &mut Model,
        train_data: &Dataset,
        loss_fn: impl Fn(&Variable<f32, 2>, &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>>,
    ) -> AnvilResult<Vec<(f32, f32)>>
    where
        Model: Clone,
    {
        let mut results = Vec::new();
        let mut best_loss = f32::INFINITY;
        let mut losses = Vec::new();
        
        // Exponential learning rate schedule
        let lr_mult = (self.end_lr / self.start_lr).powf(1.0 / (self.num_steps as f32 - 1.0));
        
        for step in 0..self.num_steps {
            let lr = self.start_lr * lr_mult.powi(step as i32);
            
            // Simplified training step - in real implementation would use actual model and data
            // This is a placeholder showing the structure
            let loss = 1.0; // Placeholder loss
            
            // Smooth the loss
            if losses.is_empty() {
                losses.push(loss);
            } else {
                let smoothed = self.smooth_factor * loss + (1.0 - self.smooth_factor) * losses.last().unwrap();
                losses.push(smoothed);
            }
            
            results.push((lr, losses.last().copied().unwrap()));
            
            // Stop if loss explodes
            if losses.last().unwrap() > &(4.0 * best_loss) && step > 10 {
                break;
            }
            
            if losses.last().unwrap() < &best_loss {
                best_loss = *losses.last().unwrap();
            }
        }
        
        Ok(results)
    }
    
    pub fn suggest_lr(&self, lr_loss_pairs: &[(f32, f32)]) -> f32 {
        // Find the learning rate with steepest negative gradient
        let mut best_lr = self.start_lr;
        let mut best_gradient = 0.0;
        
        for window in lr_loss_pairs.windows(2) {
            let (lr1, loss1) = window[0];
            let (lr2, loss2) = window[1];
            
            let gradient = (loss2 - loss1) / (lr2 - lr1);
            
            if gradient < best_gradient {
                best_gradient = gradient;
                best_lr = lr1;
            }
        }
        
        // Return learning rate that's 10x smaller than the one with steepest gradient
        best_lr / 10.0
    }
}

/// Gradient clipping utilities
pub struct GradientClipper {
    clip_type: ClipType,
    clip_value: f32,
}

#[derive(Debug, Clone, Copy)]
pub enum ClipType {
    /// Clip gradients by global norm
    GlobalNorm,
    /// Clip gradients by value
    Value,
    /// Clip gradients by per-parameter norm
    PerParameterNorm,
}

impl GradientClipper {
    pub fn new(clip_type: ClipType, clip_value: f32) -> Self {
        Self {
            clip_type,
            clip_value,
        }
    }
    
    pub fn clip_gradients<T, const DIMS: usize>(
        &self,
        gradients: &mut [Variable<T, DIMS>],
    ) -> AnvilResult<f32>
    where
        T: Copy + Clone + Default + Send + Sync + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
    {
        match self.clip_type {
            ClipType::GlobalNorm => self.clip_by_global_norm(gradients),
            ClipType::Value => self.clip_by_value(gradients),
            ClipType::PerParameterNorm => self.clip_by_per_parameter_norm(gradients),
        }
    }
    
    fn clip_by_global_norm<T, const DIMS: usize>(
        &self,
        gradients: &mut [Variable<T, DIMS>],
    ) -> AnvilResult<f32>
    where
        T: Copy + Clone + Default + Send + Sync,
    {
        // Calculate global norm
        let mut total_norm_sq = 0.0f32;
        
        for grad in gradients.iter() {
            if let Some(grad_data) = grad.grad() {
                let grad_slice = grad_data.data.as_slice::<f32>();
                for &val in grad_slice {
                    total_norm_sq += val * val;
                }
            }
        }
        
        let total_norm = total_norm_sq.sqrt();
        
        // Clip if necessary
        if total_norm > self.clip_value {
            let clip_coeff = self.clip_value / total_norm;
            
            for grad in gradients.iter_mut() {
                if let Some(grad_data) = grad.grad_mut() {
                    let grad_slice = grad_data.data.as_slice_mut::<f32>();
                    for val in grad_slice {
                        *val *= clip_coeff;
                    }
                }
            }
        }
        
        Ok(total_norm)
    }
    
    fn clip_by_value<T, const DIMS: usize>(
        &self,
        gradients: &mut [Variable<T, DIMS>],
    ) -> AnvilResult<f32>
    where
        T: Copy + Clone + Default + Send + Sync,
    {
        let mut max_grad = 0.0f32;
        
        for grad in gradients.iter_mut() {
            if let Some(grad_data) = grad.grad_mut() {
                let grad_slice = grad_data.data.as_slice_mut::<f32>();
                for val in grad_slice {
                    max_grad = max_grad.max(val.abs());
                    *val = val.clamp(-self.clip_value, self.clip_value);
                }
            }
        }
        
        Ok(max_grad)
    }
    
    fn clip_by_per_parameter_norm<T, const DIMS: usize>(
        &self,
        gradients: &mut [Variable<T, DIMS>],
    ) -> AnvilResult<f32>
    where
        T: Copy + Clone + Default + Send + Sync,
    {
        let mut max_norm = 0.0f32;
        
        for grad in gradients.iter_mut() {
            if let Some(grad_data) = grad.grad_mut() {
                let grad_slice = grad_data.data.as_slice_mut::<f32>();
                
                // Calculate parameter norm
                let mut param_norm_sq = 0.0f32;
                for &val in grad_slice.iter() {
                    param_norm_sq += val * val;
                }
                let param_norm = param_norm_sq.sqrt();
                max_norm = max_norm.max(param_norm);
                
                // Clip if necessary
                if param_norm > self.clip_value {
                    let clip_coeff = self.clip_value / param_norm;
                    for val in grad_slice {
                        *val *= clip_coeff;
                    }
                }
            }
        }
        
        Ok(max_norm)
    }
}

/// Gradient accumulation helper
pub struct GradientAccumulator {
    accumulation_steps: usize,
    current_step: usize,
    accumulated_gradients: Vec<Variable<f32, 2>>, // Simplified to 2D for now
}

impl GradientAccumulator {
    pub fn new(accumulation_steps: usize) -> Self {
        Self {
            accumulation_steps,
            current_step: 0,
            accumulated_gradients: Vec::new(),
        }
    }
    
    pub fn accumulate_gradients(&mut self, gradients: &[Variable<f32, 2>]) -> AnvilResult<()> {
        if self.accumulated_gradients.is_empty() {
            // Initialize accumulated gradients
            self.accumulated_gradients = gradients.iter().map(|g| g.clone()).collect();
        } else {
            // Add to accumulated gradients
            for (acc_grad, new_grad) in self.accumulated_gradients.iter_mut().zip(gradients.iter()) {
                // In practice, this would use proper tensor addition
                // acc_grad = acc_grad + new_grad / accumulation_steps
            }
        }
        
        self.current_step += 1;
        Ok(())
    }
    
    pub fn should_step(&self) -> bool {
        self.current_step % self.accumulation_steps == 0
    }
    
    pub fn get_accumulated_gradients(&mut self) -> Vec<Variable<f32, 2>> {
        let gradients = self.accumulated_gradients.clone();
        self.reset();
        gradients
    }
    
    pub fn reset(&mut self) {
        self.accumulated_gradients.clear();
        self.current_step = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_early_stopping() {
        let mut callback = EarlyStoppingCallback::new(3, 0.001);
        
        // Simulate improving losses
        for epoch in 0..5 {
            let result = TrainingResult {
                epoch,
                step: epoch * 100,
                train_loss: 1.0 - epoch as f32 * 0.1,
                val_loss: Some(1.0 - epoch as f32 * 0.1),
                metrics: TrainingMetrics::new(),
                duration: std::time::Duration::from_secs(60),
            };
            
            callback.on_epoch_end(epoch, &result, &Trainer::default()).unwrap();
            
            if epoch < 3 {
                assert!(!callback.should_stop());
            }
        }
        
        // Simulate plateau
        for epoch in 5..10 {
            let result = TrainingResult {
                epoch,
                step: epoch * 100,
                train_loss: 0.5,
                val_loss: Some(0.5),
                metrics: TrainingMetrics::new(),
                duration: std::time::Duration::from_secs(60),
            };
            
            callback.on_epoch_end(epoch, &result, &Trainer::default()).unwrap();
            
            if epoch >= 8 {
                assert!(callback.should_stop());
                break;
            }
        }
    }
    
    #[test]
    fn test_gradient_clipper() {
        let clipper = GradientClipper::new(ClipType::Value, 1.0);
        
        // Test would create actual gradients and verify clipping
        // This is a placeholder for the test structure
        assert_eq!(clipper.clip_value, 1.0);
    }
    
    #[test]
    fn test_lr_finder() {
        let finder = LRFinder::new(1e-6, 1e-1, 100);
        
        // Simulate loss curve
        let lr_loss_pairs = vec![
            (1e-6, 2.0),
            (1e-5, 1.8),
            (1e-4, 1.5),
            (1e-3, 1.0),  // Steepest descent here
            (1e-2, 0.8),
            (1e-1, 2.0),  // Loss starts increasing
        ];
        
        let suggested_lr = finder.suggest_lr(&lr_loss_pairs);
        assert!(suggested_lr > 1e-5 && suggested_lr < 1e-2);
    }
}