//! Neural network layers

use async_trait::async_trait;
use crate::{
    error::{AnvilError, AnvilResult},
    tensor::{AdvancedTensor, Shape, DType, Device},
    ops::core::{AdvancedTensorOperation, TensorOperation},
    nn::optimization::BatchNormOp,
};
use crate::ops::optimization::DropoutOp;
use crate::ops::conv::MaxPool2dOp;

/// Dense layer with advanced features
#[derive(Debug, Clone)]
pub struct DenseLayer {
    linear: crate::ops::linear::LinearOp,
    activation: Option<crate::ops::activation::ReLUOp>,
    dropout: Option<DropoutOp>,
    batch_norm: Option<BatchNormOp>,
    optimization_level: crate::ops::core::OptimizationLevel,
}

impl DenseLayer {
    pub fn new(
        input_size: usize,
        output_size: usize,
        activation: bool,
        dropout_rate: Option<f32>,
        batch_norm: bool,
    ) -> Self {
        let linear = crate::ops::linear::LinearOp::new(input_size, output_size);
        let activation = if activation {
            Some(crate::ops::activation::ReLUOp::new())
        } else {
            None
        };
        let dropout = dropout_rate.map(|rate| DropoutOp::new(rate, true));
        let batch_norm = if batch_norm {
            Some(BatchNormOp::new(output_size))
        } else {
            None
        };
        
        Self {
            linear,
            activation,
            dropout,
            batch_norm,
            optimization_level: crate::ops::core::OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: crate::ops::core::OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
}

#[async_trait]
impl TensorOperation<2> for DenseLayer {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        let mut output = self.linear.forward(input).await?;
        
        if let Some(ref activation) = self.activation {
            output = crate::ops::core::TensorOperation::forward(activation, &output).await?;
        }
        
        if let Some(ref dropout) = self.dropout {
            output = crate::ops::core::TensorOperation::forward(dropout, &output).await?;
        }
        
        if let Some(ref batch_norm) = self.batch_norm {
            output = crate::ops::core::TensorOperation::forward(batch_norm, &output).await?;
        }
        
        Ok(output)
    }
}

/// Convolutional layer with advanced features
#[derive(Debug, Clone)]
pub struct ConvLayer {
    conv: crate::ops::conv::Conv2dOp,
    activation: Option<crate::ops::activation::ReLUOp>,
    batch_norm: Option<BatchNormOp>,
    pooling: Option<MaxPool2dOp>,
    optimization_level: crate::ops::core::OptimizationLevel,
}

impl ConvLayer {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        activation: bool,
        batch_norm: bool,
        pooling: bool,
    ) -> Self {
        // Create weight tensor for convolution
        let weight_shape = Shape::new([out_channels, in_channels, kernel_size, kernel_size]);
        let weight_tensor = AdvancedTensor::<f32, 4>::new(weight_shape, DType::F32, Device::Cpu).unwrap();
        
        // Create bias tensor for convolution
        let bias_shape = Shape::new([out_channels]);
        let bias_tensor = AdvancedTensor::<f32, 1>::new(bias_shape, DType::F32, Device::Cpu).unwrap();
        
        let conv = crate::ops::conv::Conv2dOp::new(
            weight_tensor,
            Some(bias_tensor),
            (stride, stride),
            (padding, padding)
        );
        let activation = if activation {
            Some(crate::ops::activation::ReLUOp::new())
        } else {
            None
        };
        let batch_norm = if batch_norm {
            Some(BatchNormOp::new(out_channels))
        } else {
            None
        };
        let pooling = if pooling {
            Some(MaxPool2dOp::new(2, 2, 0))
        } else {
            None
        };
        
        Self {
            conv,
            activation,
            batch_norm,
            pooling,
            optimization_level: crate::ops::core::OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: crate::ops::core::OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
}

#[async_trait]
impl TensorOperation<4> for ConvLayer {
    async fn forward(&self, input: &AdvancedTensor<f32, 4>) -> AnvilResult<AdvancedTensor<f32, 4>> {
        let mut output = self.conv.forward(input).await?;
        
        if let Some(ref activation) = self.activation {
            output = crate::ops::core::TensorOperation::forward(activation, &output).await?;
        }
        
        if let Some(ref batch_norm) = self.batch_norm {
            output = crate::ops::core::TensorOperation::forward(batch_norm, &output).await?;
        }
        
        if let Some(ref pooling) = self.pooling {
            output = crate::ops::core::TensorOperation::forward(pooling, &output).await?;
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_dense_layer() {
        let input = AdvancedTensor::<f32, 2>::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap();
        let dense = DenseLayer::new(3, 4, true, Some(0.5), true);
        let result = dense.forward(&input).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_conv_layer() {
        let input = AdvancedTensor::<f32, 4>::new(Shape::new([1, 3, 32, 32]), DType::F32, Device::Cpu).unwrap();
        let conv = ConvLayer::new(3, 64, 3, 1, 1, true, true, true);
        let result = conv.forward(&input).await;
        assert!(result.is_ok());
    }
} 