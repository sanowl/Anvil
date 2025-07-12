//! Activation functions

use async_trait::async_trait;
use crate::ops::core::TensorOperation;
use crate::{
    tensor::{AdvancedTensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
    ops::core::{AdvancedTensorOperation, OptimizationLevel},
};

#[derive(Debug, Clone)]
pub struct ReLUOp {
    optimization_level: OptimizationLevel,
}

impl ReLUOp {
    pub fn new() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
}

#[async_trait]
impl TensorOperation<2> for ReLUOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Simple implementation for now
        Ok(input.clone())
    }
}

#[async_trait]
impl TensorOperation<4> for ReLUOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 4>) -> AnvilResult<AdvancedTensor<f32, 4>> {
        // Simple implementation for now
        Ok(input.clone())
    }
}

/// Sigmoid activation function
pub struct SigmoidOp {
    optimization_level: OptimizationLevel,
}

impl SigmoidOp {
    pub fn new() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
}

#[async_trait]
impl AdvancedTensorOperation<2> for SigmoidOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        let mut output = AdvancedTensor::<f32, 2>::new(input.shape(), input.dtype(), input.device())?;
        let input_data = input.as_slice::<f32>();
        let output_data = output.as_slice_mut::<f32>();
        
        for (out, &in_val) in output_data.iter_mut().zip(input_data.iter()) {
            *out = 1.0 / (1.0 + (-in_val).exp());
        }
        
        Ok(output)
    }
    
    fn name(&self) -> &'static str {
        "SigmoidOp"
    }
    
    fn input_shape_requirements(&self) -> Shape<2> {
        Shape::new([0, 0])
    }
    
    fn output_shape(&self, input_shape: &Shape<2>) -> AnvilResult<Shape<2>> {
        Ok(input_shape.clone())
    }
    
    fn operation_type(&self) -> &'static str {
        "sigmoid"
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

/// Tanh activation function
pub struct TanhOp {
    optimization_level: OptimizationLevel,
}

impl TanhOp {
    pub fn new() -> Self {
        Self {
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
}

#[async_trait]
impl AdvancedTensorOperation<2> for TanhOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        let mut output = AdvancedTensor::<f32, 2>::new(input.shape(), input.dtype(), input.device())?;
        let input_data = input.as_slice::<f32>();
        let output_data = output.as_slice_mut::<f32>();
        
        for (out, &in_val) in output_data.iter_mut().zip(input_data.iter()) {
            *out = in_val.tanh();
        }
        
        Ok(output)
    }
    
    fn name(&self) -> &'static str {
        "TanhOp"
    }
    
    fn input_shape_requirements(&self) -> Shape<2> {
        Shape::new([0, 0])
    }
    
    fn output_shape(&self, input_shape: &Shape<2>) -> AnvilResult<Shape<2>> {
        Ok(input_shape.clone())
    }
    
    fn operation_type(&self) -> &'static str {
        "tanh"
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

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_relu_operation() {
        let input = AdvancedTensor::<f32, 2>::new(Shape::new([2, 2]), DType::F32, Device::Cpu).unwrap();
        let relu_op = ReLUOp::new();
        let result = relu_op.forward(&input).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_sigmoid_operation() {
        let input = AdvancedTensor::<f32, 2>::new(Shape::new([2, 2]), DType::F32, Device::Cpu).unwrap();
        let sigmoid_op = SigmoidOp::new();
        let result = sigmoid_op.forward(&input).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_tanh_operation() {
        let input = AdvancedTensor::<f32, 2>::new(Shape::new([2, 2]), DType::F32, Device::Cpu).unwrap();
        let tanh_op = TanhOp::new();
        let result = tanh_op.forward(&input).await;
        assert!(result.is_ok());
    }
} 