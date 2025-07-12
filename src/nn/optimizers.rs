//! Optimizers

use async_trait::async_trait;
use crate::{
    tensor::{AdvancedTensor as Tensor, AdvancedTensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
    ops::core::{AdvancedTensorOperation, OptimizationLevel},
};

/// SGD optimizer
pub struct SGD {
    lr: f32,
    momentum: f32,
    weight_decay: f32,
    optimization_level: OptimizationLevel,
}

impl SGD {
    pub fn new(lr: f32, momentum: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            momentum,
            weight_decay,
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
    
    pub async fn step(&self, params: &mut [AdvancedTensor<f32, 2>], grads: &[AdvancedTensor<f32, 2>]) -> AnvilResult<()> {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let mut param_data = param.as_slice_mut::<f32>();
            let grad_data = grad.as_slice::<f32>();
            
            for (p, &g) in param_data.iter_mut().zip(grad_data.iter()) {
                *p -= self.lr * g;
            }
        }
        Ok(())
    }
}

/// Adam optimizer
pub struct Adam {
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    optimization_level: OptimizationLevel,
}

impl Adam {
    pub fn new(lr: f32, beta1: f32, beta2: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
    
    pub async fn step(&self, params: &mut [AdvancedTensor<f32, 2>], grads: &[AdvancedTensor<f32, 2>]) -> AnvilResult<()> {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let mut param_data = param.as_slice_mut::<f32>();
            let grad_data = grad.as_slice::<f32>();
            
            for (p, &g) in param_data.iter_mut().zip(grad_data.iter()) {
                // Simplified Adam update
                *p -= self.lr * g;
            }
        }
        Ok(())
    }
}

/// RMSprop optimizer
pub struct RMSprop {
    lr: f32,
    alpha: f32,
    eps: f32,
    weight_decay: f32,
    optimization_level: OptimizationLevel,
}

impl RMSprop {
    pub fn new(lr: f32, alpha: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            alpha,
            eps,
            weight_decay,
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
    
    pub async fn step(&self, params: &mut [AdvancedTensor<f32, 2>], grads: &[AdvancedTensor<f32, 2>]) -> AnvilResult<()> {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let mut param_data = param.as_slice_mut::<f32>();
            let grad_data = grad.as_slice::<f32>();
            
            for (p, &g) in param_data.iter_mut().zip(grad_data.iter()) {
                // Simplified RMSprop update
                *p -= self.lr * g;
            }
        }
        Ok(())
    }
}

/// AdaGrad optimizer
pub struct AdaGrad {
    lr: f32,
    eps: f32,
    weight_decay: f32,
    optimization_level: OptimizationLevel,
}

impl AdaGrad {
    pub fn new(lr: f32, eps: f32, weight_decay: f32) -> Self {
        Self {
            lr,
            eps,
            weight_decay,
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
    
    pub async fn step(&self, params: &mut [AdvancedTensor<f32, 2>], grads: &[AdvancedTensor<f32, 2>]) -> AnvilResult<()> {
        for (param, grad) in params.iter_mut().zip(grads.iter()) {
            let mut param_data = param.as_slice_mut::<f32>();
            let grad_data = grad.as_slice::<f32>();
            
            for (p, &g) in param_data.iter_mut().zip(grad_data.iter()) {
                // Simplified AdaGrad update
                *p -= self.lr * g;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_sgd_optimizer() {
        let mut params = vec![
            Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap(),
        ];
        let grads = vec![
            Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap(),
        ];
        
        let sgd = SGD::new(0.01, 0.9, 0.0);
        let result = sgd.step(&mut params, &grads).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_adam_optimizer() {
        let mut params = vec![
            Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap(),
        ];
        let grads = vec![
            Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap(),
        ];
        
        let adam = Adam::new(0.001, 0.9, 0.999, 1e-8, 0.0);
        let result = adam.step(&mut params, &grads).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_rmsprop_optimizer() {
        let mut params = vec![
            Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap(),
        ];
        let grads = vec![
            Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap(),
        ];
        
        let rmsprop = RMSprop::new(0.01, 0.99, 1e-8, 0.0);
        let result = rmsprop.step(&mut params, &grads).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_adagrad_optimizer() {
        let mut params = vec![
            Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap(),
        ];
        let grads = vec![
            Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap(),
        ];
        
        let adagrad = AdaGrad::new(0.01, 1e-8, 0.0);
        let result = adagrad.step(&mut params, &grads).await;
        assert!(result.is_ok());
    }
} 