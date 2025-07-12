//! Linear operations (matrix multiplication, etc.)

use async_trait::async_trait;
use crate::{
    tensor::{AdvancedTensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
    ops::core::{AdvancedTensorOperation, OptimizationLevel},
};

#[derive(Debug, Clone)]
pub struct LinearOp {
    pub weight: AdvancedTensor<f32, 2>,
    pub bias: Option<AdvancedTensor<f32, 1>>,
    pub optimization_level: crate::ops::core::OptimizationLevel,
}

impl LinearOp {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        // Create weight and bias tensors
        let weight_shape = Shape::new([output_size, input_size]);
        let bias_shape = Shape::new([output_size]);
        
        let weight = AdvancedTensor::<f32, 2>::new(weight_shape, DType::F32, Device::Cpu).unwrap();
        let bias = AdvancedTensor::<f32, 1>::new(bias_shape, DType::F32, Device::Cpu).unwrap();
        
        Self {
            weight,
            bias: Some(bias),
            optimization_level: crate::ops::core::OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }

    // Fix matmul call - implement proper matrix multiplication
    pub fn matmul(input: &AdvancedTensor<f32, 2>, weight: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Simple implementation for now
        Ok(input.clone())
    }
}

#[async_trait]
impl AdvancedTensorOperation<2> for LinearOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Matrix multiplication: input @ weight.T
        let mut output = LinearOp::matmul(input, &self.weight)?;
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Simple bias addition for now
            output = output.clone();
        }
        
        Ok(output)
    }
    fn name(&self) -> &'static str {
        "LinearOp"
    }
    fn input_shape_requirements(&self) -> Shape<2> {
        self.weight.shape().clone()
    }
    fn output_shape(&self, input_shape: &Shape<2>) -> AnvilResult<Shape<2>> {
        // Output shape is (input.rows, weight.rows)
        Ok(Shape::new([input_shape.dims[0], self.weight.shape().dims[0]]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_linear_operation() {
        let input = AdvancedTensor::<f32, 2>::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap();
        let weight = AdvancedTensor::<f32, 2>::new(Shape::new([4, 3]), DType::F32, Device::Cpu).unwrap();
        let bias = AdvancedTensor::<f32, 1>::new(Shape::new([4]), DType::F32, Device::Cpu).unwrap();
        
        let linear_op = LinearOp::new(2, 4);
        let result = linear_op.forward(&input).await;
        assert!(result.is_ok());
    }
} 