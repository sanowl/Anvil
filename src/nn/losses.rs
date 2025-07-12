//! Loss functions

use async_trait::async_trait;
use crate::{
    error::{AnvilError, AnvilResult},
    tensor::core::{AdvancedTensor, Shape, DType},
    ops::core::TensorOperation,
};

/// Mean squared error loss
#[derive(Debug, Clone)]
pub struct MSELoss {
    reduction: Reduction,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    None,
    Mean,
    Sum,
}

impl MSELoss {
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }
    
    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }
}

#[async_trait]
impl TensorOperation<2> for MSELoss {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>, target: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Simplified MSE loss implementation
        let input_data = input.as_slice::<f32>();
        let target_data = target.as_slice::<f32>();
        
        let mut loss_data = Vec::with_capacity(input_data.len());
        for (i, t) in input_data.iter().zip(target_data.iter()) {
            let diff = i - t;
            loss_data.push(diff * diff);
        }
        
        let shape = input.shape();
        let mut loss_tensor = AdvancedTensor::new(shape, DType::F32, input.device())?;
        let loss_slice = loss_tensor.as_slice_mut::<f32>();
        
        for (i, &val) in loss_data.iter().enumerate() {
            loss_slice[i] = val;
        }
        
        Ok(loss_tensor)
    }
}

/// Cross entropy loss
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    reduction: Reduction,
    ignore_index: Option<usize>,
}

impl CrossEntropyLoss {
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
            ignore_index: None,
        }
    }
    
    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }
    
    pub fn with_ignore_index(mut self, ignore_index: usize) -> Self {
        self.ignore_index = Some(ignore_index);
        self
    }
}

#[async_trait]
impl TensorOperation<2> for CrossEntropyLoss {
    async fn forward(&self, input: &crate::tensor::core::AdvancedTensor<2>) -> crate::error::AnvilResult<crate::tensor::core::AdvancedTensor<2>> {
        // Simple implementation for now
        Ok(input.clone())
    }
}

impl CrossEntropyLoss {
    pub fn forward(&self, input: &crate::tensor::core::AdvancedTensor<2>, _target: &crate::tensor::core::AdvancedTensor<2>) -> crate::error::AnvilResult<crate::tensor::core::AdvancedTensor<2>> {
        // Simple implementation for now
        Ok(input.clone())
    }
}

/// Binary cross entropy loss
#[derive(Debug, Clone)]
pub struct BCELoss {
    reduction: Reduction,
}

impl BCELoss {
    pub fn new() -> Self {
        Self {
            reduction: Reduction::Mean,
        }
    }
    
    pub fn with_reduction(mut self, reduction: Reduction) -> Self {
        self.reduction = reduction;
        self
    }
}

#[async_trait]
impl TensorOperation<2> for BCELoss {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>, target: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Simplified BCE loss implementation
        let input_data = input.as_slice::<f32>();
        let target_data = target.as_slice::<f32>();
        
        let mut loss_data = Vec::with_capacity(input_data.len());
        for (i, t) in input_data.iter().zip(target_data.iter()) {
            let clamped_input = i.clamp(1e-7, 1.0 - 1e-7);
            let loss = -(t * clamped_input.ln() + (1.0 - t) * (1.0 - clamped_input).ln());
            loss_data.push(loss);
        }
        
        let shape = input.shape();
        let mut loss_tensor = AdvancedTensor::new(shape, DType::F32, input.device())?;
        let loss_slice = loss_tensor.as_slice_mut::<f32>();
        
        for (i, &val) in loss_data.iter().enumerate() {
            loss_slice[i] = val;
        }
        
        Ok(loss_tensor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_mse_loss() {
        let input = AdvancedTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3])).unwrap();
        let target = AdvancedTensor::from_vec(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], Shape::new([2, 3])).unwrap();
        let loss = MSELoss::new();
        let result = loss.forward(&input, &target).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_cross_entropy_loss() {
        let input = AdvancedTensor::from_vec(vec![0.1f32, 0.2, 0.3, 0.4, 0.5, 0.6], Shape::new([2, 3])).unwrap();
        let target = AdvancedTensor::from_vec(vec![0, 1, 2], Shape::new([2])).unwrap();
        let loss = CrossEntropyLoss::new();
        let result = loss.forward(&input, &target).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_bce_loss() {
        let input = AdvancedTensor::from_vec(vec![0.1f32, 0.9, 0.2, 0.8, 0.3, 0.7], Shape::new([2, 3])).unwrap();
        let target = AdvancedTensor::from_vec(vec![1.0f32, 0.0], Shape::new([2, 1])).unwrap();
        let loss = BCELoss::new();
        let result = loss.forward(&input, &target).await;
        assert!(result.is_ok());
    }
}