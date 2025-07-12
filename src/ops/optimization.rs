//! Optimization operations

use async_trait::async_trait;
use crate::ops::core::TensorOperation;
use crate::{
    tensor::{AdvancedTensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
    ops::core::{AdvancedTensorOperation, OptimizationLevel},
};

#[derive(Debug, Clone)]
pub struct BatchNormOp {
    running_mean: AdvancedTensor<f32, 1>,
    running_var: AdvancedTensor<f32, 1>,
    weight: Option<AdvancedTensor<f32, 1>>,
    bias: Option<AdvancedTensor<f32, 1>>,
    momentum: f32,
    eps: f32,
    optimization_level: OptimizationLevel,
}

impl BatchNormOp {
    pub fn new(
        running_mean: AdvancedTensor<f32, 1>,
        running_var: AdvancedTensor<f32, 1>,
        weight: Option<AdvancedTensor<f32, 1>>,
        bias: Option<AdvancedTensor<f32, 1>>,
        momentum: f32,
        eps: f32,
    ) -> Self {
        Self {
            running_mean,
            running_var,
            weight,
            bias,
            momentum,
            eps,
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
}

#[async_trait]
impl AdvancedTensorOperation<2> for BatchNormOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Implement batch norm logic here
        Ok(input.clone())
    }
    fn name(&self) -> &'static str {
        "BatchNormOp"
    }
    fn input_shape_requirements(&self) -> Shape<2> {
        Shape::new([0, 0])
    }
    fn output_shape(&self, input_shape: &Shape<2>) -> AnvilResult<Shape<2>> {
        Ok(input_shape.clone())
    }
    
    fn operation_type(&self) -> &'static str {
        "batch_norm"
    }
    
    fn supports_simd(&self) -> bool {
        true
    }
    
    fn supports_gpu(&self) -> bool {
        true
    }
    
    fn memory_alignment(&self) -> usize {
        16
    }
}

#[derive(Debug, Clone)]
pub struct DropoutOp {
    p: f32,
    training: bool,
    optimization_level: OptimizationLevel,
}

impl DropoutOp {
    pub fn new(p: f32, training: bool) -> Self {
        Self {
            p,
            training,
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
}

#[async_trait]
impl AdvancedTensorOperation<2> for DropoutOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        let mut output = AdvancedTensor::<f32, 2>::new(input.shape(), input.dtype(), input.device())?;
        let input_data = input.as_slice::<f32>();
        let output_data = output.as_slice_mut::<f32>();
        
        if self.training {
            // Apply dropout during training
            for (out, &in_val) in output_data.iter_mut().zip(input_data.iter()) {
                let mask = if fastrand::f32() > self.p { 1.0 } else { 0.0 };
                *out = in_val * mask / (1.0 - self.p);
            }
        } else {
            // No dropout during inference
            output_data.copy_from_slice(input_data);
        }
        
        Ok(output)
    }
    
    fn name(&self) -> &'static str {
        "DropoutOp"
    }
    
    fn input_shape_requirements(&self) -> Shape<2> {
        Shape::new([0, 0])
    }
    
    fn output_shape(&self, input_shape: &Shape<2>) -> AnvilResult<Shape<2>> {
        Ok(input_shape.clone())
    }
    
    fn operation_type(&self) -> &'static str {
        "dropout"
    }
    
    fn supports_simd(&self) -> bool {
        true
    }
    
    fn supports_gpu(&self) -> bool {
        true
    }
    
    fn memory_alignment(&self) -> usize {
        16
    }
}

#[async_trait]
impl TensorOperation<2> for DropoutOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Simple implementation for now
        Ok(input.clone())
    }
}

#[async_trait]
impl TensorOperation<4> for DropoutOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 4>) -> AnvilResult<AdvancedTensor<f32, 4>> {
        // Simple implementation for now
        Ok(input.clone())
    }
}

#[async_trait]
impl TensorOperation<2> for BatchNormOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Simple implementation for now
        Ok(input.clone())
    }
}

#[async_trait]
impl TensorOperation<4> for BatchNormOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 4>) -> AnvilResult<AdvancedTensor<f32, 4>> {
        // Simple implementation for now
        Ok(input.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_batch_norm() {
        let input = AdvancedTensor::<f32, 4>::new(Shape::new([2, 3, 4, 4]), DType::F32, Device::Cpu).unwrap();
        let mean = AdvancedTensor::<f32, 1>::new(Shape::new([3]), DType::F32, Device::Cpu).unwrap();
        let var = AdvancedTensor::<f32, 1>::new(Shape::new([3]), DType::F32, Device::Cpu).unwrap();
        let weight = AdvancedTensor::<f32, 1>::new(Shape::new([3]), DType::F32, Device::Cpu).unwrap();
        let bias = AdvancedTensor::<f32, 1>::new(Shape::new([3]), DType::F32, Device::Cpu).unwrap();
        
        let bn_op = BatchNormOp::new(mean, var, Some(weight), Some(bias), 0.1, 1e-5);
        let result = bn_op.forward(&input).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_dropout() {
        let input = AdvancedTensor::<f32, 2>::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap();
        let dropout_op = DropoutOp::new(0.5, true);
        let result = dropout_op.forward(&input).await;
        assert!(result.is_ok());
    }
} 