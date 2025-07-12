//! Convolution operations with optimized algorithms

use async_trait::async_trait;
use rayon::prelude::*;
use std::sync::Arc;
use crate::{
    tensor::{AdvancedTensor as Tensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
    ops::core::{AdvancedTensorOperation, OptimizationLevel, TensorOperation},
};

#[derive(Debug, Clone)]
pub struct Conv2dOp {
    weight: Tensor<f32, 4>,
    bias: Option<Tensor<f32, 1>>,
    stride: (usize, usize),
    padding: (usize, usize),
    optimization_level: OptimizationLevel,
}

impl Conv2dOp {
    pub fn new(
        weight: Tensor<f32, 4>,
        bias: Option<Tensor<f32, 1>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> Self {
        Self {
            weight,
            bias,
            stride,
            padding,
            optimization_level: OptimizationLevel::Basic,
        }
    }
    
    pub fn with_optimization(mut self, level: OptimizationLevel) -> Self {
        self.optimization_level = level;
        self
    }
    
    /// High-performance 2D convolution with multiple algorithms
    pub fn conv2d_optimized(
        input: &Tensor<f32, 4>,
        weight: &Tensor<f32, 4>,
        bias: Option<&Tensor<f32, 1>>,
        stride: (usize, usize),
        padding: (usize, usize),
        algorithm: ConvAlgorithm,
    ) -> AnvilResult<Tensor<f32, 4>> {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        
        // Input: [N, C, H, W], Weight: [F, C, KH, KW]
        let batch_size = input_shape.dims[0];
        let in_channels = input_shape.dims[1];
        let in_height = input_shape.dims[2];
        let in_width = input_shape.dims[3];
        
        let out_channels = weight_shape.dims[0];
        let kernel_height = weight_shape.dims[2];
        let kernel_width = weight_shape.dims[3];
        
        // Validate input/weight compatibility
        if weight_shape.dims[1] != in_channels {
            return Err(AnvilError::ComputationError(
                format!("Channel mismatch: input {} vs weight {}", in_channels, weight_shape.dims[1])
            ));
        }
        
        // Calculate output dimensions
        let out_height = (in_height + 2 * padding.0 - kernel_height) / stride.0 + 1;
        let out_width = (in_width + 2 * padding.1 - kernel_width) / stride.1 + 1;
        
        let output_shape = Shape::new([batch_size, out_channels, out_height, out_width]);
        let mut output = Tensor::new(output_shape, DType::F32, input.device())?;
        
        match algorithm {
            ConvAlgorithm::Im2Col => {
                Self::conv2d_im2col(input, weight, &mut output, stride, padding)?;
            },
            ConvAlgorithm::Direct => {
                Self::conv2d_direct(input, weight, &mut output, stride, padding)?;
            },
            ConvAlgorithm::Winograd => {
                if kernel_height == 3 && kernel_width == 3 && stride == (1, 1) {
                    Self::conv2d_winograd_3x3(input, weight, &mut output, padding)?;
                } else {
                    // Fallback to im2col for unsupported Winograd cases
                    Self::conv2d_im2col(input, weight, &mut output, stride, padding)?;
                }
            },
        }
        
        // Add bias if present
        if let Some(bias) = bias {
            Self::add_bias(&mut output, bias)?;
        }
        
        Ok(output)
    }
    
    /// Im2Col-based convolution (GEMM approach)
    fn conv2d_im2col(
        input: &Tensor<f32, 4>,
        weight: &Tensor<f32, 4>,
        output: &mut Tensor<f32, 4>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> AnvilResult<()> {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let output_shape = output.shape();
        
        let batch_size = input_shape.dims[0];
        let in_channels = input_shape.dims[1];
        let in_height = input_shape.dims[2];
        let in_width = input_shape.dims[3];
        
        let out_channels = weight_shape.dims[0];
        let kernel_height = weight_shape.dims[2];
        let kernel_width = weight_shape.dims[3];
        let out_height = output_shape.dims[2];
        let out_width = output_shape.dims[3];
        
        let input_data = input.as_slice::<f32>();
        let weight_data = weight.as_slice::<f32>();
        let output_data = output.as_slice_mut::<f32>();
        
        // Create im2col matrix: [kernel_size * in_channels, out_height * out_width]
        let col_size = kernel_height * kernel_width * in_channels;
        let output_size = out_height * out_width;
        
        (0..batch_size).into_par_iter().for_each(|batch| {
            let mut col_buffer = vec![0.0f32; col_size * output_size];
            
            // Im2col transformation
            Self::im2col_cpu(
                &input_data[batch * in_channels * in_height * in_width..],
                &mut col_buffer,
                in_channels, in_height, in_width,
                kernel_height, kernel_width,
                padding, stride,
                out_height, out_width,
            );
            
            // GEMM: weight @ col_buffer -> output
            for out_c in 0..out_channels {
                for out_idx in 0..output_size {
                    let mut sum = 0.0f32;
                    for k in 0..col_size {
                        sum += weight_data[out_c * col_size + k] * col_buffer[k * output_size + out_idx];
                    }
                    let output_idx = batch * out_channels * out_height * out_width +
                                   out_c * out_height * out_width + out_idx;
                    output_data[output_idx] = sum;
                }
            }
        });
        
        Ok(())
    }
    
    /// Direct convolution implementation
    fn conv2d_direct(
        input: &Tensor<f32, 4>,
        weight: &Tensor<f32, 4>,
        output: &mut Tensor<f32, 4>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> AnvilResult<()> {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        let output_shape = output.shape();
        
        let batch_size = input_shape.dims[0];
        let in_channels = input_shape.dims[1];
        let in_height = input_shape.dims[2];
        let in_width = input_shape.dims[3];
        
        let out_channels = weight_shape.dims[0];
        let kernel_height = weight_shape.dims[2];
        let kernel_width = weight_shape.dims[3];
        let out_height = output_shape.dims[2];
        let out_width = output_shape.dims[3];
        
        let input_data = input.as_slice::<f32>();
        let weight_data = weight.as_slice::<f32>();
        let output_data = output.as_slice_mut::<f32>();
        
        (0..batch_size).into_par_iter().for_each(|batch| {
            for out_c in 0..out_channels {
                for out_h in 0..out_height {
                    for out_w in 0..out_width {
                        let mut sum = 0.0f32;
                        
                        for in_c in 0..in_channels {
                            for kh in 0..kernel_height {
                                for kw in 0..kernel_width {
                                    let in_h = out_h * stride.0 + kh;
                                    let in_w = out_w * stride.1 + kw;
                                    
                                    // Apply padding
                                    if in_h >= padding.0 && in_w >= padding.1 &&
                                       in_h < in_height + padding.0 && in_w < in_width + padding.1 {
                                        let actual_h = in_h - padding.0;
                                        let actual_w = in_w - padding.1;
                                        
                                        if actual_h < in_height && actual_w < in_width {
                                            let input_idx = batch * in_channels * in_height * in_width +
                                                           in_c * in_height * in_width +
                                                           actual_h * in_width + actual_w;
                                            let weight_idx = out_c * in_channels * kernel_height * kernel_width +
                                                           in_c * kernel_height * kernel_width +
                                                           kh * kernel_width + kw;
                                            
                                            sum += input_data[input_idx] * weight_data[weight_idx];
                                        }
                                    }
                                }
                            }
                        }
                        
                        let output_idx = batch * out_channels * out_height * out_width +
                                       out_c * out_height * out_width +
                                       out_h * out_width + out_w;
                        output_data[output_idx] = sum;
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Winograd convolution for 3x3 kernels (F(2x2, 3x3))
    fn conv2d_winograd_3x3(
        input: &Tensor<f32, 4>,
        weight: &Tensor<f32, 4>,
        output: &mut Tensor<f32, 4>,
        padding: (usize, usize),
    ) -> AnvilResult<()> {
        // Winograd F(2,3) transformation matrices
        let g = [
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.0, 0.0, 1.0],
        ];
        
        let bt = [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, -1.0],
        ];
        
        let at = [
            [1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, -1.0, -1.0],
        ];
        
        // For simplicity, fallback to direct convolution
        // Full Winograd implementation would require extensive matrix transformations
        Self::conv2d_direct(input, weight, output, (1, 1), padding)
    }
    
    /// Im2col CPU implementation
    fn im2col_cpu(
        input: &[f32],
        col: &mut [f32],
        channels: usize, height: usize, width: usize,
        kernel_h: usize, kernel_w: usize,
        padding: (usize, usize), stride: (usize, usize),
        out_h: usize, out_w: usize,
    ) {
        for c in 0..channels {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let kernel_idx = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    
                    for out_y in 0..out_h {
                        for out_x in 0..out_w {
                            let in_y = out_y * stride.0 + kh;
                            let in_x = out_x * stride.1 + kw;
                            
                            let col_idx = kernel_idx * out_h * out_w + out_y * out_w + out_x;
                            
                            if in_y >= padding.0 && in_x >= padding.1 &&
                               in_y < height + padding.0 && in_x < width + padding.1 {
                                let actual_y = in_y - padding.0;
                                let actual_x = in_x - padding.1;
                                
                                if actual_y < height && actual_x < width {
                                    let input_idx = c * height * width + actual_y * width + actual_x;
                                    col[col_idx] = input[input_idx];
                                } else {
                                    col[col_idx] = 0.0;
                                }
                            } else {
                                col[col_idx] = 0.0;
                            }
                        }
                    }
                }
            }
        }
    }
    
    /// Add bias to convolution output
    fn add_bias(output: &mut Tensor<f32, 4>, bias: &Tensor<f32, 1>) -> AnvilResult<()> {
        let output_shape = output.shape();
        let batch_size = output_shape.dims[0];
        let out_channels = output_shape.dims[1];
        let out_height = output_shape.dims[2];
        let out_width = output_shape.dims[3];
        
        let output_data = output.as_slice_mut::<f32>();
        let bias_data = bias.as_slice::<f32>();
        
        (0..batch_size).into_par_iter().for_each(|batch| {
            for c in 0..out_channels {
                let channel_bias = bias_data[c];
                let start_idx = batch * out_channels * out_height * out_width +
                              c * out_height * out_width;
                let end_idx = start_idx + out_height * out_width;
                
                for idx in start_idx..end_idx {
                    output_data[idx] += channel_bias;
                }
            }
        });
        
        Ok(())
    }
}

#[async_trait]
impl AdvancedTensorOperation<4> for Conv2dOp {
    async fn forward(&self, input: &Tensor<f32, 4>) -> AnvilResult<Tensor<f32, 4>> {
        // Choose algorithm based on optimization level and kernel size
        let algorithm = match self.optimization_level {
            OptimizationLevel::Maximum => {
                let kernel_size = self.weight.shape().dims[2];
                if kernel_size == 3 && self.stride == (1, 1) {
                    ConvAlgorithm::Winograd
                } else {
                    ConvAlgorithm::Im2Col
                }
            },
            OptimizationLevel::Moderate => ConvAlgorithm::Im2Col,
            _ => ConvAlgorithm::Direct,
        };
        
        Conv2dOp::conv2d_optimized(
            input,
            &self.weight,
            self.bias.as_ref(),
            self.stride,
            self.padding,
            algorithm,
        )
    }
    
    fn name(&self) -> &'static str {
        "Conv2dOp"
    }
    
    fn input_shape_requirements(&self) -> Shape<4> {
        // Require 4D tensor [N, C, H, W]
        Shape::new([1, self.weight.shape().dims[1], 1, 1])
    }
    
    fn output_shape(&self, input_shape: &Shape<4>) -> AnvilResult<Shape<4>> {
        let weight_shape = self.weight.shape();
        let out_height = (input_shape.dims[2] + 2 * self.padding.0 - weight_shape.dims[2]) / self.stride.0 + 1;
        let out_width = (input_shape.dims[3] + 2 * self.padding.1 - weight_shape.dims[3]) / self.stride.1 + 1;
        
        Ok(Shape::new([
            input_shape.dims[0], // batch size
            weight_shape.dims[0], // output channels
            out_height,
            out_width,
        ]))
    }
}

/// Convolution algorithms for different optimization scenarios
#[derive(Debug, Clone, Copy)]
pub enum ConvAlgorithm {
    /// Direct convolution (good for small kernels)
    Direct,
    /// Im2col + GEMM (good for large kernels)
    Im2Col,
    /// Winograd algorithm (good for 3x3 kernels with stride 1)
    Winograd,
}

#[derive(Debug, Clone)]
pub struct MaxPool2dOp {
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
}

impl MaxPool2dOp {
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        Self { kernel_size: (kernel_size, kernel_size), stride: (stride, stride), padding: (padding, padding) }
    }
}

#[async_trait]
impl TensorOperation<4> for MaxPool2dOp {
    async fn forward(&self, input: &crate::tensor::core::AdvancedTensor<f32, 4>) -> crate::error::AnvilResult<crate::tensor::core::AdvancedTensor<f32, 4>> {
        let input_shape = input.shape();
        let batch_size = input_shape.dims[0];
        let channels = input_shape.dims[1];
        let in_height = input_shape.dims[2];
        let in_width = input_shape.dims[3];
        
        // Calculate output dimensions
        let out_height = (in_height + 2 * self.padding.0 - self.kernel_size.0) / self.stride.0 + 1;
        let out_width = (in_width + 2 * self.padding.1 - self.kernel_size.1) / self.stride.1 + 1;
        
        let output_shape = Shape::new([batch_size, channels, out_height, out_width]);
        let mut output = crate::tensor::core::AdvancedTensor::new(output_shape, DType::F32, input.device())?;
        
        let input_data = input.as_slice::<f32>();
        let output_data = output.as_slice_mut::<f32>();
        
        // Parallel max pooling
        (0..batch_size).into_par_iter().for_each(|batch| {
            for c in 0..channels {
                for out_h in 0..out_height {
                    for out_w in 0..out_width {
                        let mut max_val = f32::NEG_INFINITY;
                        
                        for kh in 0..self.kernel_size.0 {
                            for kw in 0..self.kernel_size.1 {
                                let in_h = out_h * self.stride.0 + kh;
                                let in_w = out_w * self.stride.1 + kw;
                                
                                if in_h >= self.padding.0 && in_w >= self.padding.1 &&
                                   in_h < in_height + self.padding.0 && in_w < in_width + self.padding.1 {
                                    let actual_h = in_h - self.padding.0;
                                    let actual_w = in_w - self.padding.1;
                                    
                                    if actual_h < in_height && actual_w < in_width {
                                        let input_idx = batch * channels * in_height * in_width +
                                                       c * in_height * in_width +
                                                       actual_h * in_width + actual_w;
                                        max_val = max_val.max(input_data[input_idx]);
                                    }
                                }
                            }
                        }
                        
                        let output_idx = batch * channels * out_height * out_width +
                                       c * out_height * out_width +
                                       out_h * out_width + out_w;
                        output_data[output_idx] = max_val;
                    }
                }
            }
        });
        
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_conv2d_operation() {
        let input = Tensor::new(Shape::new([1, 3, 32, 32]), DType::F32, Device::Cpu).unwrap();
        let weight = Tensor::new(Shape::new([64, 3, 3, 3]), DType::F32, Device::Cpu).unwrap();
        let bias = Tensor::new(Shape::new([64]), DType::F32, Device::Cpu).unwrap();
        
        let conv_op = Conv2dOp::new(weight, Some(bias), (1, 1), (1, 1));
        let result = conv_op.forward(&input).await;
        assert!(result.is_ok());
    }
} 