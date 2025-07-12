//! Linear operations (matrix multiplication, etc.)

use async_trait::async_trait;
use std::sync::Arc;
use rayon::prelude::*;
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

    /// High-performance matrix multiplication with BLAS-style optimizations
    pub fn matmul(input: &AdvancedTensor<f32, 2>, weight: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        let input_shape = input.shape();
        let weight_shape = weight.shape();
        
        // Validate dimensions for matrix multiplication
        if input_shape.dims[1] != weight_shape.dims[1] {
            return Err(AnvilError::ComputationError(
                format!("Matrix multiplication dimension mismatch: {}x{} @ {}x{}", 
                    input_shape.dims[0], input_shape.dims[1],
                    weight_shape.dims[0], weight_shape.dims[1])
            ));
        }
        
        let m = input_shape.dims[0];  // Input rows
        let k = input_shape.dims[1];  // Inner dimension
        let n = weight_shape.dims[0]; // Output columns
        
        let output_shape = Shape::new([m, n]);
        let mut output = AdvancedTensor::new(output_shape, DType::F32, input.device())?;
        
        let input_data = input.as_slice::<f32>();
        let weight_data = weight.as_slice::<f32>();
        let output_data = output.as_slice_mut::<f32>();
        
        // Choose optimal algorithm based on matrix size
        if m * k * n > 1000000 { // Large matrices
            Self::matmul_blocked_parallel(input_data, weight_data, output_data, m, k, n);
        } else if m * k * n > 10000 { // Medium matrices
            Self::matmul_parallel(input_data, weight_data, output_data, m, k, n);
        } else { // Small matrices
            Self::matmul_sequential(input_data, weight_data, output_data, m, k, n);
        }
        
        Ok(output)
    }
    
    /// Cache-efficient blocked matrix multiplication with parallelization
    fn matmul_blocked_parallel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        const BLOCK_SIZE: usize = 64; // Optimized for L1 cache
        
        (0..m).into_par_iter().step_by(BLOCK_SIZE).for_each(|i_block| {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                for k_block in (0..k).step_by(BLOCK_SIZE) {
                    let i_end = (i_block + BLOCK_SIZE).min(m);
                    let j_end = (j_block + BLOCK_SIZE).min(n);
                    let k_end = (k_block + BLOCK_SIZE).min(k);
                    
                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0f32;
                            for kk in k_block..k_end {
                                sum += a[i * k + kk] * b[j * k + kk];
                            }
                            unsafe {
                                let c_ptr = c.as_mut_ptr().add(i * n + j);
                                *c_ptr += sum;
                            }
                        }
                    }
                }
            }
        });
    }
    
    /// Parallel matrix multiplication for medium-sized matrices
    fn matmul_parallel(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        (0..m).into_par_iter().for_each(|i| {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[j * k + kk];
                }
                c[i * n + j] = sum;
            }
        });
    }
    
    /// Sequential matrix multiplication for small matrices
    fn matmul_sequential(a: &[f32], b: &[f32], c: &mut [f32], m: usize, k: usize, n: usize) {
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for kk in 0..k {
                    sum += a[i * k + kk] * b[j * k + kk];
                }
                c[i * n + j] = sum;
            }
        }
    }
    
    /// Element-wise tensor addition with broadcasting support
    pub fn add(a: &AdvancedTensor<f32, 2>, b: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        
        // Simple case: same shapes
        if a_shape.dims == b_shape.dims {
            let mut output = AdvancedTensor::new(a_shape.clone(), DType::F32, a.device())?;
            let a_data = a.as_slice::<f32>();
            let b_data = b.as_slice::<f32>();
            let output_data = output.as_slice_mut::<f32>();
            
            output_data.par_iter_mut()
                .zip(a_data.par_iter().zip(b_data.par_iter()))
                .for_each(|(out, (&a_val, &b_val))| {
                    *out = a_val + b_val;
                });
            
            Ok(output)
        } else {
            // Implement broadcasting logic
            Self::add_broadcast(a, b)
        }
    }
    
    /// Broadcasting addition for tensors with different shapes
    fn add_broadcast(a: &AdvancedTensor<f32, 2>, b: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        
        // Determine output shape (broadcast to larger dimensions)
        let output_shape = Shape::new([
            a_shape.dims[0].max(b_shape.dims[0]),
            a_shape.dims[1].max(b_shape.dims[1]),
        ]);
        
        let mut output = AdvancedTensor::new(output_shape.clone(), DType::F32, a.device())?;
        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let output_data = output.as_slice_mut::<f32>();
        
        let out_rows = output_shape.dims[0];
        let out_cols = output_shape.dims[1];
        
        (0..out_rows).into_par_iter().for_each(|i| {
            for j in 0..out_cols {
                let a_i = if a_shape.dims[0] == 1 { 0 } else { i };
                let a_j = if a_shape.dims[1] == 1 { 0 } else { j };
                let b_i = if b_shape.dims[0] == 1 { 0 } else { i };
                let b_j = if b_shape.dims[1] == 1 { 0 } else { j };
                
                let a_val = a_data[a_i * a_shape.dims[1] + a_j];
                let b_val = b_data[b_i * b_shape.dims[1] + b_j];
                output_data[i * out_cols + j] = a_val + b_val;
            }
        });
        
        Ok(output)
    }
    
    /// Element-wise tensor multiplication
    pub fn mul(a: &AdvancedTensor<f32, 2>, b: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        let a_shape = a.shape();
        let b_shape = b.shape();
        
        if a_shape.dims != b_shape.dims {
            return Err(AnvilError::ComputationError(
                "Element-wise multiplication requires same shapes".to_string()
            ));
        }
        
        let mut output = AdvancedTensor::new(a_shape.clone(), DType::F32, a.device())?;
        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();
        let output_data = output.as_slice_mut::<f32>();
        
        output_data.par_iter_mut()
            .zip(a_data.par_iter().zip(b_data.par_iter()))
            .for_each(|(out, (&a_val, &b_val))| {
                *out = a_val * b_val;
            });
        
        Ok(output)
    }
}

#[async_trait]
impl AdvancedTensorOperation<2> for LinearOp {
    async fn forward(&self, input: &AdvancedTensor<f32, 2>) -> AnvilResult<AdvancedTensor<f32, 2>> {
        // Matrix multiplication: input @ weight.T (weight is stored transposed)
        let mut output = LinearOp::matmul(input, &self.weight)?;
        
        // Add bias if present
        if let Some(ref bias) = self.bias {
            // Broadcast bias across batch dimension
            let bias_data = bias.as_slice::<f32>();
            let output_data = output.as_slice_mut::<f32>();
            let output_shape = output.shape();
            
            (0..output_shape.dims[0]).into_par_iter().for_each(|i| {
                for j in 0..output_shape.dims[1] {
                    let idx = i * output_shape.dims[1] + j;
                    unsafe {
                        let output_ptr = output_data.as_mut_ptr().add(idx);
                        *output_ptr += bias_data[j];
                    }
                }
            });
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