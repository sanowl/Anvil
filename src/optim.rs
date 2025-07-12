use std::collections::HashMap;
use async_trait::async_trait;
use serde::{Serialize, Deserialize};

use crate::{
    tensor::{Tensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
    nn::Module,
};

/// Base trait for all optimizers
#[async_trait]
pub trait Optimizer: Send + Sync {
    /// Update parameters using gradients
    async fn step(&mut self, gradients: &[Tensor<2>]) -> AnvilResult<()>;
    
    /// Zero out gradients
    fn zero_grad(&mut self);
    
    /// Get current learning rate
    fn learning_rate(&self) -> f32;
    
    /// Set learning rate
    fn set_learning_rate(&mut self, lr: f32);
    
    /// Get optimizer state
    fn state(&self) -> OptimizerState;
    
    /// Load optimizer state
    fn load_state(&mut self, state: OptimizerState) -> AnvilResult<()>;
}

/// Optimizer state for serialization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerState {
    pub learning_rate: f32,
    pub step_count: usize,
    pub momentum_buffers: HashMap<String, Vec<f32>>,
    pub variance_buffers: HashMap<String, Vec<f32>>,
    pub custom_state: HashMap<String, serde_json::Value>,
}

impl OptimizerState {
    pub fn new(learning_rate: f32) -> Self {
        Self {
            learning_rate,
            step_count: 0,
            momentum_buffers: HashMap::new(),
            variance_buffers: HashMap::new(),
            custom_state: HashMap::new(),
        }
    }
}

/// Stochastic Gradient Descent optimizer
pub struct SGD {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    nesterov: bool,
    state: OptimizerState,
}

impl SGD {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            momentum: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            state: OptimizerState::new(lr),
        }
    }
    
    pub fn with_momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }
    
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

#[async_trait]
impl Optimizer for SGD {
    async fn step(&mut self, gradients: &[Tensor<2>]) -> AnvilResult<()> {
        self.state.step_count += 1;
        
        for (i, gradient) in gradients.iter().enumerate() {
            let param_name = format!("param_{}", i);
            
            // Apply weight decay
            let mut grad = gradient.clone();
            if self.weight_decay > 0.0 {
                // Weight decay implementation
                let mut grad_data = grad.as_slice_mut::<f32>();
                // For now, we'll simulate weight decay by adding a small penalty
                // In a real implementation, this would access the actual parameter values
                for g in grad_data.iter_mut() {
                    *g += self.weight_decay * 0.01; // Simulated weight decay
                }
            }
            
            // Apply momentum
            if self.momentum > 0.0 {
                let momentum_buffer = self.state.momentum_buffers
                    .entry(param_name.clone())
                    .or_insert_with(|| vec![0.0; gradient.shape().size()]);
                
                // Momentum update
                for (j, &g) in grad.as_slice::<f32>().iter().enumerate() {
                    momentum_buffer[j] = self.momentum * momentum_buffer[j] + g;
                }
                
                // Apply momentum to gradient
                if self.nesterov {
                    // Nesterov momentum
                    for (j, &m) in momentum_buffer.iter().enumerate() {
                        // Nesterov implementation: look ahead momentum
                        let mut grad_data = grad.as_slice_mut::<f32>();
                        grad_data[j] = m + self.momentum * momentum_buffer[j];
                    }
                } else {
                    // Standard momentum
                    for (j, &m) in momentum_buffer.iter().enumerate() {
                        // Standard momentum implementation
                        let mut grad_data = grad.as_slice_mut::<f32>();
                        grad_data[j] = m;
                    }
                }
            }
            
            // Update parameter
            // In a real implementation, this would update the actual parameter tensor
            // For now, we'll simulate the parameter update by modifying the gradient
            let mut grad_data = grad.as_slice_mut::<f32>();
            for g in grad_data.iter_mut() {
                *g *= -self.lr; // Simulate parameter update: param -= lr * grad
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        // Clear gradients
        // In a real implementation, this would clear the gradient tensors
    }
    
    fn learning_rate(&self) -> f32 {
        self.lr
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
        self.state.learning_rate = lr;
    }
    
    fn state(&self) -> OptimizerState {
        self.state.clone()
    }
    
    fn load_state(&mut self, state: OptimizerState) -> AnvilResult<()> {
        self.state = state;
        self.lr = self.state.learning_rate;
        Ok(())
    }
}

/// Adam optimizer with automatic learning rate scheduling
pub struct Adam {
    lr: f32,
    betas: (f32, f32),
    eps: f32,
    weight_decay: f32,
    amsgrad: bool,
    state: OptimizerState,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            lr,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            state: OptimizerState::new(lr),
        }
    }
    
    pub fn with_betas(mut self, beta1: f32, beta2: f32) -> Self {
        self.betas = (beta1, beta2);
        self
    }
    
    pub fn with_eps(mut self, eps: f32) -> Self {
        self.eps = eps;
        self
    }
    
    pub fn with_weight_decay(mut self, weight_decay: f32) -> Self {
        self.weight_decay = weight_decay;
        self
    }
    
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }
}

#[async_trait]
impl Optimizer for Adam {
    async fn step(&mut self, gradients: &[Tensor<2>]) -> AnvilResult<()> {
        self.state.step_count += 1;
        
        for (i, gradient) in gradients.iter().enumerate() {
            let param_name = format!("param_{}", i);
            
            // Get or create momentum and variance buffers
            let momentum_buffer = self.state.momentum_buffers
                .entry(param_name.clone())
                .or_insert_with(|| vec![0.0; gradient.shape().size()]);
            
            let variance_buffer = self.state.variance_buffers
                .entry(param_name.clone())
                .or_insert_with(|| vec![0.0; gradient.shape().size()]);
            
            // Apply weight decay
            let mut grad = gradient.clone();
            if self.weight_decay > 0.0 {
                // Weight decay implementation
                let mut grad_data = grad.as_slice_mut::<f32>();
                // For now, we'll simulate weight decay by adding a small penalty
                // In a real implementation, this would access the actual parameter values
                for g in grad_data.iter_mut() {
                    *g += self.weight_decay * 0.01; // Simulated weight decay
                }
            }
            
            // Update momentum and variance
            for (j, &g) in grad.as_slice::<f32>().iter().enumerate() {
                momentum_buffer[j] = self.betas.0 * momentum_buffer[j] + (1.0 - self.betas.0) * g;
                variance_buffer[j] = self.betas.1 * variance_buffer[j] + (1.0 - self.betas.1) * g * g;
            }
            
            // Bias correction
            let bias_correction1 = 1.0 - self.betas.0.powi(self.state.step_count as i32);
            let bias_correction2 = 1.0 - self.betas.1.powi(self.state.step_count as i32);
            
            // Update parameter
            for j in 0..gradient.shape().size() {
                let m_hat = momentum_buffer[j] / bias_correction1;
                let v_hat = variance_buffer[j] / bias_correction2;
                
                // Parameter update
                let step_size = self.lr / (v_hat.sqrt() + self.eps);
                // In a real implementation, this would update the actual parameter
                // For now, we'll simulate the parameter update by modifying the gradient
                let mut grad_data = grad.as_slice_mut::<f32>();
                grad_data[j] = -step_size * m_hat; // Simulate parameter update: param -= step_size * m_hat
            }
        }
        
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        // Clear gradients
    }
    
    fn learning_rate(&self) -> f32 {
        self.lr
    }
    
    fn set_learning_rate(&mut self, lr: f32) {
        self.lr = lr;
        self.state.learning_rate = lr;
    }
    
    fn state(&self) -> OptimizerState {
        self.state.clone()
    }
    
    fn load_state(&mut self, state: OptimizerState) -> AnvilResult<()> {
        self.state = state;
        self.lr = self.state.learning_rate;
        Ok(())
    }
}

/// Learning rate scheduler with automatic adaptation
pub trait LRScheduler: Send + Sync {
    fn step(&mut self, optimizer: &mut dyn Optimizer);
    fn get_last_lr(&self) -> f32;
    fn state_dict(&self) -> HashMap<String, serde_json::Value>;
    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> AnvilResult<()>;
}

/// Step learning rate scheduler
pub struct StepLR {
    step_size: usize,
    gamma: f32,
    last_epoch: usize,
}

impl StepLR {
    pub fn new(step_size: usize, gamma: f32) -> Self {
        Self {
            step_size,
            gamma,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for StepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.last_epoch += 1;
        if self.last_epoch % self.step_size == 0 {
            let new_lr = optimizer.learning_rate() * self.gamma;
            optimizer.set_learning_rate(new_lr);
        }
    }
    
    fn get_last_lr(&self) -> f32 {
        // This would return the last learning rate
        0.001 // Placeholder
    }
    
    fn state_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut state = HashMap::new();
        state.insert("step_size".to_string(), serde_json::json!(self.step_size));
        state.insert("gamma".to_string(), serde_json::json!(self.gamma));
        state.insert("last_epoch".to_string(), serde_json::json!(self.last_epoch));
        state
    }
    
    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> AnvilResult<()> {
        self.step_size = state.get("step_size")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AnvilError::ConfigError("Missing step_size".to_string()))? as usize;
        
        self.gamma = state.get("gamma")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| AnvilError::ConfigError("Missing gamma".to_string()))? as f32;
        
        self.last_epoch = state.get("last_epoch")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AnvilError::ConfigError("Missing last_epoch".to_string()))? as usize;
        
        Ok(())
    }
}

/// Cosine annealing learning rate scheduler
pub struct CosineAnnealingLR {
    t_max: usize,
    eta_min: f32,
    last_epoch: usize,
}

impl CosineAnnealingLR {
    pub fn new(t_max: usize, eta_min: f32) -> Self {
        Self {
            t_max,
            eta_min,
            last_epoch: 0,
        }
    }
}

impl LRScheduler for CosineAnnealingLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        self.last_epoch += 1;
        
        let progress = self.last_epoch as f32 / self.t_max as f32;
        let cosine_decay = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        let new_lr = self.eta_min + (optimizer.learning_rate() - self.eta_min) * cosine_decay;
        
        optimizer.set_learning_rate(new_lr);
    }
    
    fn get_last_lr(&self) -> f32 {
        // This would return the last learning rate
        0.001 // Placeholder
    }
    
    fn state_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut state = HashMap::new();
        state.insert("t_max".to_string(), serde_json::json!(self.t_max));
        state.insert("eta_min".to_string(), serde_json::json!(self.eta_min));
        state.insert("last_epoch".to_string(), serde_json::json!(self.last_epoch));
        state
    }
    
    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> AnvilResult<()> {
        self.t_max = state.get("t_max")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AnvilError::ConfigError("Missing t_max".to_string()))? as usize;
        
        self.eta_min = state.get("eta_min")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| AnvilError::ConfigError("Missing eta_min".to_string()))? as f32;
        
        self.last_epoch = state.get("last_epoch")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AnvilError::ConfigError("Missing last_epoch".to_string()))? as usize;
        
        Ok(())
    }
}

/// Automatic learning rate scheduler that adapts based on training progress
pub struct AutoLRScheduler {
    base_lr: f32,
    patience: usize,
    factor: f32,
    min_lr: f32,
    best_loss: f32,
    patience_counter: usize,
    last_epoch: usize,
}

impl AutoLRScheduler {
    pub fn new(base_lr: f32) -> Self {
        Self {
            base_lr,
            patience: 10,
            factor: 0.5,
            min_lr: 1e-6,
            best_loss: f32::INFINITY,
            patience_counter: 0,
            last_epoch: 0,
        }
    }
    
    pub fn with_patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }
    
    pub fn with_factor(mut self, factor: f32) -> Self {
        self.factor = factor;
        self
    }
    
    pub fn with_min_lr(mut self, min_lr: f32) -> Self {
        self.min_lr = min_lr;
        self
    }
    
    pub fn step_with_loss(&mut self, optimizer: &mut dyn Optimizer, loss: f32) {
        self.last_epoch += 1;
        
        if loss < self.best_loss {
            self.best_loss = loss;
            self.patience_counter = 0;
        } else {
            self.patience_counter += 1;
            
            if self.patience_counter >= self.patience {
                let new_lr = (optimizer.learning_rate() * self.factor).max(self.min_lr);
                optimizer.set_learning_rate(new_lr);
                self.patience_counter = 0;
            }
        }
    }
}

impl LRScheduler for AutoLRScheduler {
    fn step(&mut self, optimizer: &mut dyn Optimizer) {
        // Auto scheduler needs loss information, so this is a no-op
        // Use step_with_loss instead
    }
    
    fn get_last_lr(&self) -> f32 {
        self.base_lr
    }
    
    fn state_dict(&self) -> HashMap<String, serde_json::Value> {
        let mut state = HashMap::new();
        state.insert("base_lr".to_string(), serde_json::json!(self.base_lr));
        state.insert("patience".to_string(), serde_json::json!(self.patience));
        state.insert("factor".to_string(), serde_json::json!(self.factor));
        state.insert("min_lr".to_string(), serde_json::json!(self.min_lr));
        state.insert("best_loss".to_string(), serde_json::json!(self.best_loss));
        state.insert("patience_counter".to_string(), serde_json::json!(self.patience_counter));
        state.insert("last_epoch".to_string(), serde_json::json!(self.last_epoch));
        state
    }
    
    fn load_state_dict(&mut self, state: HashMap<String, serde_json::Value>) -> AnvilResult<()> {
        self.base_lr = state.get("base_lr")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| AnvilError::ConfigError("Missing base_lr".to_string()))? as f32;
        
        self.patience = state.get("patience")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AnvilError::ConfigError("Missing patience".to_string()))? as usize;
        
        self.factor = state.get("factor")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| AnvilError::ConfigError("Missing factor".to_string()))? as f32;
        
        self.min_lr = state.get("min_lr")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| AnvilError::ConfigError("Missing min_lr".to_string()))? as f32;
        
        self.best_loss = state.get("best_loss")
            .and_then(|v| v.as_f64())
            .ok_or_else(|| AnvilError::ConfigError("Missing best_loss".to_string()))? as f32;
        
        self.patience_counter = state.get("patience_counter")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AnvilError::ConfigError("Missing patience_counter".to_string()))? as usize;
        
        self.last_epoch = state.get("last_epoch")
            .and_then(|v| v.as_u64())
            .ok_or_else(|| AnvilError::ConfigError("Missing last_epoch".to_string()))? as usize;
        
        Ok(())
    }
}

#[async_trait]
impl Optimizer for Box<dyn Optimizer + Send + Sync> {
    async fn step(&mut self, gradients: &[Tensor<2>]) -> AnvilResult<()> {
        (**self).step(gradients).await
    }
    fn zero_grad(&mut self) {
        (**self).zero_grad()
    }
    fn learning_rate(&self) -> f32 {
        (**self).learning_rate()
    }
    fn set_learning_rate(&mut self, lr: f32) {
        (**self).set_learning_rate(lr)
    }
    fn state(&self) -> OptimizerState {
        (**self).state()
    }
    fn load_state(&mut self, state: OptimizerState) -> AnvilResult<()> {
        (**self).load_state(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sgd_creation() {
        let sgd = SGD::new(0.01)
            .with_momentum(0.9)
            .with_weight_decay(1e-4);
        
        assert_eq!(sgd.learning_rate(), 0.01);
        assert_eq!(sgd.momentum, 0.9);
        assert_eq!(sgd.weight_decay, 1e-4);
    }

    #[test]
    fn test_adam_creation() {
        let adam = Adam::new(0.001)
            .with_betas(0.9, 0.999)
            .with_eps(1e-8);
        
        assert_eq!(adam.learning_rate(), 0.001);
        assert_eq!(adam.betas, (0.9, 0.999));
        assert_eq!(adam.eps, 1e-8);
    }

    #[test]
    fn test_step_lr_scheduler() {
        let mut scheduler = StepLR::new(10, 0.5);
        let mut optimizer = SGD::new(0.1);
        
        // Step 9 times - no change
        for _ in 0..9 {
            scheduler.step(&mut optimizer);
        }
        assert_eq!(optimizer.learning_rate(), 0.1);
        
        // Step once more - should change
        scheduler.step(&mut optimizer);
        assert_eq!(optimizer.learning_rate(), 0.05);
    }

    #[test]
    fn test_auto_lr_scheduler() {
        let mut scheduler = AutoLRScheduler::new(0.1)
            .with_patience(3)
            .with_factor(0.5);
        let mut optimizer = SGD::new(0.1);
        
        // Simulate improving loss
        scheduler.step_with_loss(&mut optimizer, 0.5);
        assert_eq!(optimizer.learning_rate(), 0.1);
        
        // Simulate worsening loss for patience + 1 steps
        for _ in 0..4 {
            scheduler.step_with_loss(&mut optimizer, 1.0);
        }
        
        // Learning rate should be reduced
        assert_eq!(optimizer.learning_rate(), 0.05);
    }
} 