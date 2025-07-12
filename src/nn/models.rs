//! Neural network models

use async_trait::async_trait;
use crate::{
    error::{AnvilError, AnvilResult},
    tensor::core::{AdvancedTensor, Shape},
    ops::core::{AdvancedTensorOperation, TensorOperation},
    nn::layers::{DenseLayer, ConvLayer},
};
use std::fmt::Debug;

/// Feed-forward neural network
#[derive(Debug)]
pub struct FeedForwardNet {
    pub layers: Vec<Box<dyn crate::ops::core::TensorOperation<2> + Send + Sync + Debug>>,
    input_size: usize,
    output_size: usize,
}

impl FeedForwardNet {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            layers: Vec::new(),
            input_size,
            output_size,
        }
    }
    
    pub fn add_dense_layer(mut self, input_size: usize, output_size: usize, activation: bool, dropout_rate: Option<f32>, batch_norm: bool) -> Self {
        let layer = DenseLayer::new(input_size, output_size, activation, dropout_rate, batch_norm);
        self.layers.push(Box::new(layer));
        self
    }
    
    pub fn build(self) -> Self {
        self
    }
}

#[async_trait]
impl TensorOperation<2> for FeedForwardNet {
    async fn forward(&self, input: &AdvancedTensor<2>) -> AnvilResult<AdvancedTensor<2>> {
        let mut output = input.clone();
        
        for layer in &self.layers {
            output = layer.forward(&output).await?;
        }
        
        Ok(output)
    }
}

/// Convolutional neural network
#[derive(Debug)]
pub struct ConvNet {
    pub layers: Vec<Box<dyn crate::ops::core::TensorOperation<4> + Send + Sync + Debug>>,
    input_channels: usize,
    num_classes: usize,
}

impl ConvNet {
    pub fn new(input_channels: usize, num_classes: usize) -> Self {
        Self {
            layers: Vec::new(),
            input_channels,
            num_classes,
        }
    }
    
    pub fn add_conv_layer(mut self, in_channels: usize, out_channels: usize, kernel_size: usize, stride: usize, padding: usize, activation: bool, batch_norm: bool, pooling: bool) -> Self {
        let layer = ConvLayer::new(in_channels, out_channels, kernel_size, stride, padding, activation, batch_norm, pooling);
        self.layers.push(Box::new(layer));
        self
    }
    
    pub fn add_dense_layer(mut self, input_size: usize, output_size: usize, activation: bool, dropout_rate: Option<f32>, batch_norm: bool) -> Self {
        let layer = DenseLayer::new(input_size, output_size, activation, dropout_rate, batch_norm);
        self.layers.push(Box::new(layer));
        self
    }
    
    pub fn build(self) -> Self {
        self
    }
}

#[async_trait]
impl TensorOperation<2> for ConvNet {
    async fn forward(&self, input: &AdvancedTensor<4>) -> AnvilResult<AdvancedTensor<2>> {
        let mut output = input.clone();
        
        for layer in &self.layers {
            // Handle dimension changes between conv and dense layers
            if output.shape().dims.len() == 4 {
                // Conv layer
                output = layer.forward(&output).await?;
            } else {
                // Dense layer - need to reshape
                let flattened_shape = Shape::new([output.shape().dims[0], output.numel() / output.shape().dims[0]]);
                output = output.reshape(flattened_shape)?;
                output = layer.forward(&output).await?;
            }
        }
        
        Ok(output)
    }
}

/// Transformer model
#[derive(Debug)]
pub struct Transformer {
    pub attention: crate::ops::attention::MultiHeadAttentionOp,
    pub feed_forward: FeedForwardNet,
    num_layers: usize,
    embed_dim: usize,
    num_heads: usize,
}

impl Transformer {
    pub fn new(embed_dim: usize, num_heads: usize, num_layers: usize, ff_dim: usize) -> Self {
        let attention = crate::ops::attention::MultiHeadAttentionOp::new(embed_dim, num_heads);
        let feed_forward = FeedForwardNet::new(embed_dim, ff_dim)
            .add_dense_layer(embed_dim, ff_dim, true, None, false)
            .add_dense_layer(ff_dim, embed_dim, false, None, false)
            .build();
        
        Self {
            attention,
            feed_forward,
            num_layers,
            embed_dim,
            num_heads,
        }
    }
}

#[async_trait]
impl TensorOperation<3> for Transformer {
    async fn forward(&self, input: &AdvancedTensor<3>) -> AnvilResult<AdvancedTensor<3>> {
        let mut output = input.clone();
        
        for _ in 0..self.num_layers {
            // Self-attention
            let attention_output = self.attention.forward(&output).await?;
            let residual1 = output.add(&attention_output)?;
            
            // Feed-forward
            let ff_output = crate::ops::core::TensorOperation::forward(&self.feed_forward, &residual1).await?;
            output = residual1.add(&ff_output)?;
        }
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_feedforward_net() {
        let input = AdvancedTensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap();
        let net = FeedForwardNet::new(3, 2);
        let result = net.forward(&input).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_conv_net() {
        let input = AdvancedTensor::new(Shape::new([1, 3, 32, 32]), DType::F32, Device::Cpu).unwrap();
        let net = ConvNet::new(3, 10);
        let result = net.forward(&input).await;
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_transformer_block() {
        let input = AdvancedTensor::new(Shape::new([10, 2, 512]), DType::F32, Device::Cpu).unwrap();
        let block = Transformer::new(512, 8, 2, 2048);
        let result = block.forward(&input).await;
        assert!(result.is_ok());
    }
} 