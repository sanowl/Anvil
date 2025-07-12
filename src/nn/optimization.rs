use async_trait::async_trait;
use crate::{
    error::{AnvilError, AnvilResult},
    tensor::core::{AdvancedTensor, Shape, DType},
    ops::core::TensorOperation,
};

#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            momentum: 0.9,
            weight_decay: 0.0,
        }
    }
}

pub trait Optimizer {
    fn step(&mut self, gradients: &[AdvancedTensor<f32, 2>]) -> AnvilResult<()>;
    fn zero_grad(&mut self);
}

pub struct SGD {
    config: OptimizationConfig,
    parameters: Vec<AdvancedTensor<f32, 2>>,
}

impl SGD {
    pub fn new(config: OptimizationConfig) -> Self {
        Self { 
            config,
            parameters: Vec::new(),
        }
    }
    
    pub fn add_parameter(&mut self, param: AdvancedTensor<f32, 2>) {
        self.parameters.push(param);
    }
}

impl Optimizer for SGD {
    fn step(&mut self, gradients: &[AdvancedTensor<f32, 2>]) -> AnvilResult<()> {
        if gradients.len() != self.parameters.len() {
            return Err(AnvilError::ComputationError(
                "Gradient and parameter count mismatch".to_string()
            ));
        }
        
        for (param, grad) in self.parameters.iter_mut().zip(gradients.iter()) {
            let param_data = param.as_slice_mut::<f32>();
            let grad_data = grad.as_slice::<f32>();
            
            if param_data.len() != grad_data.len() {
                return Err(AnvilError::ComputationError(
                    "Parameter and gradient shape mismatch".to_string()
                ));
            }
            
            for (p, &g) in param_data.iter_mut().zip(grad_data.iter()) {
                let weight_decay_term = if self.config.weight_decay > 0.0 {
                    self.config.weight_decay * *p
                } else {
                    0.0
                };
                *p -= self.config.learning_rate * (g + weight_decay_term);
            }
        }
        Ok(())
    }
    
    fn zero_grad(&mut self) {
        for param in &mut self.parameters {
            let param_data = param.as_slice_mut::<f32>();
            for p in param_data.iter_mut() {
                *p = 0.0;
            }
        }
    }
}

/// Batch normalization operation
#[derive(Debug, Clone)]
pub struct BatchNormOp {
    num_features: usize,
    eps: f32,
    momentum: f32,
    running_mean: Vec<f32>,
    running_var: Vec<f32>,
    weight: Vec<f32>,
    bias: Vec<f32>,
    training: bool,
}

impl BatchNormOp {
    pub fn new(num_features: usize) -> Self {
        Self {
            num_features,
            eps: 1e-5,
            momentum: 0.1,
            running_mean: vec![0.0; num_features],
            running_var: vec![1.0; num_features],
            weight: vec![1.0; num_features],
            bias: vec![0.0; num_features],
            training: true,
        }
    }
    
    pub fn with_optimization(mut self, _level: crate::ops::core::OptimizationLevel) -> Self {
        // Apply optimization level
        self
    }
    
    pub fn eval(&mut self) {
        self.training = false;
    }
    
    pub fn train(&mut self) {
        self.training = true;
    }
}

impl TensorOperation<2> for BatchNormOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Simplified batch normalization
        let input_data = input.as_slice::<f32>();
        let shape = input.shape();
        let mut output = AdvancedTensor::new(shape, DType::F32, input.device())?;
        let output_data = output.as_slice_mut::<f32>();
        
        // Simple normalization: (x - mean) / std
        let mean = input_data.iter().sum::<f32>() / input_data.len() as f32;
        let variance = input_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / input_data.len() as f32;
        let std = variance.sqrt().max(1e-8);
        
        for (i, &val) in input_data.iter().enumerate() {
            output_data[i] = (val - mean) / std;
        }
        
        Ok(output)
    }
}

/// Dropout operation
#[derive(Debug, Clone)]
pub struct DropoutOp {
    rate: f32,
    training: bool,
}

impl DropoutOp {
    pub fn new(rate: f32) -> Self {
        Self {
            rate,
            training: true,
        }
    }
    
    pub fn eval(&mut self) {
        self.training = false;
    }
    
    pub fn train(&mut self) {
        self.training = true;
    }
}

impl TensorOperation<2> for DropoutOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Simplified dropout implementation
        let input_data = input.as_slice::<f32>();
        let shape = input.shape();
        let mut output = AdvancedTensor::new(shape, DType::F32, input.device())?;
        let output_data = output.as_slice_mut::<f32>();
        
        if self.training {
            // Training mode: apply dropout
            for (i, &val) in input_data.iter().enumerate() {
                let mask = if fastrand::f32() < self.rate {
                    1.0
                } else {
                    0.0
                };
                output_data[i] = val * mask / (1.0 - self.rate);
            }
        } else {
            // Inference mode: no dropout
            for (i, &val) in input_data.iter().enumerate() {
                output_data[i] = val;
            }
        }
        
        Ok(output)
    }
} 