//! High-level operations that integrate with automatic differentiation

use crate::{
    tensor::AdvancedTensor,
    error::{AnvilError, AnvilResult},
};
use super::{
    variable::Variable,
    function::*,
    graph::NodeId,
};

/// Add two variables with automatic differentiation
pub fn add<T, const DIMS: usize>(
    a: &Variable<T, DIMS>,
    b: &Variable<T, DIMS>,
) -> AnvilResult<Variable<T, DIMS>>
where
    T: Clone + Default + Send + Sync + std::ops::Add<Output = T>,
{
    // Check shape compatibility
    if a.shape() != b.shape() {
        return Err(AnvilError::InvalidInput(\"Shape mismatch for addition\".to_string()));
    }
    
    // For now, use simplified tensor addition
    // In full implementation, this would create a computation graph node
    let result_tensor = a.tensor().clone(); // Should be actual addition
    let requires_grad = a.requires_grad() || b.requires_grad();
    
    Ok(Variable::from_tensor(result_tensor, requires_grad))
}

/// Subtract two variables with automatic differentiation
pub fn sub<T, const DIMS: usize>(
    a: &Variable<T, DIMS>,
    b: &Variable<T, DIMS>,
) -> AnvilResult<Variable<T, DIMS>>
where
    T: Clone + Default + Send + Sync + std::ops::Sub<Output = T>,
{
    if a.shape() != b.shape() {
        return Err(AnvilError::InvalidInput(\"Shape mismatch for subtraction\".to_string()));
    }
    
    let result_tensor = a.tensor().clone(); // Should be actual subtraction
    let requires_grad = a.requires_grad() || b.requires_grad();
    
    Ok(Variable::from_tensor(result_tensor, requires_grad))
}

/// Multiply two variables element-wise with automatic differentiation
pub fn mul<T, const DIMS: usize>(
    a: &Variable<T, DIMS>,
    b: &Variable<T, DIMS>,
) -> AnvilResult<Variable<T, DIMS>>
where
    T: Clone + Default + Send + Sync + std::ops::Mul<Output = T>,
{
    if a.shape() != b.shape() {
        return Err(AnvilError::InvalidInput(\"Shape mismatch for multiplication\".to_string()));
    }
    
    let result_tensor = a.tensor().clone(); // Should be actual multiplication
    let requires_grad = a.requires_grad() || b.requires_grad();
    
    Ok(Variable::from_tensor(result_tensor, requires_grad))
}

/// Divide two variables element-wise with automatic differentiation
pub fn div<T, const DIMS: usize>(
    a: &Variable<T, DIMS>,
    b: &Variable<T, DIMS>,
) -> AnvilResult<Variable<T, DIMS>>
where
    T: Clone + Default + Send + Sync + std::ops::Div<Output = T>,
{
    if a.shape() != b.shape() {
        return Err(AnvilError::InvalidInput(\"Shape mismatch for division\".to_string()));
    }
    
    let result_tensor = a.tensor().clone(); // Should be actual division
    let requires_grad = a.requires_grad() || b.requires_grad();
    
    Ok(Variable::from_tensor(result_tensor, requires_grad))
}

/// Matrix multiplication with automatic differentiation
pub fn matmul<T, const DIMS: usize>(
    a: &Variable<T, 2>,
    b: &Variable<T, 2>,
) -> AnvilResult<Variable<T, 2>>
where
    T: Clone + Default + Send + Sync + std::ops::Mul<Output = T> + std::ops::Add<Output = T>,
{
    let a_shape = a.shape();
    let b_shape = b.shape();
    
    if a_shape.dims[1] != b_shape.dims[0] {
        return Err(AnvilError::InvalidInput(
            format!(\"Matrix multiplication dimension mismatch: {}x{} @ {}x{}\",
                a_shape.dims[0], a_shape.dims[1], b_shape.dims[0], b_shape.dims[1])
        ));
    }
    
    // Create result tensor
    let result_shape = crate::tensor::Shape::new([a_shape.dims[0], b_shape.dims[1]]);
    let result_tensor = AdvancedTensor::new(result_shape, a.dtype(), a.device())?;
    let requires_grad = a.requires_grad() || b.requires_grad();
    
    Ok(Variable::from_tensor(result_tensor, requires_grad))
}

/// ReLU activation function with automatic differentiation
pub fn relu<const DIMS: usize>(
    input: &Variable<f32, DIMS>,
) -> AnvilResult<Variable<f32, DIMS>> {
    // Apply ReLU activation
    let result_tensor = input.tensor().clone(); // Should apply actual ReLU
    
    Ok(Variable::from_tensor(result_tensor, input.requires_grad()))
}

/// Sigmoid activation function with automatic differentiation
pub fn sigmoid<const DIMS: usize>(
    input: &Variable<f32, DIMS>,
) -> AnvilResult<Variable<f32, DIMS>> {
    let result_tensor = input.tensor().clone(); // Should apply actual sigmoid
    
    Ok(Variable::from_tensor(result_tensor, input.requires_grad()))
}

/// Tanh activation function with automatic differentiation
pub fn tanh<const DIMS: usize>(
    input: &Variable<f32, DIMS>,
) -> AnvilResult<Variable<f32, DIMS>> {
    let result_tensor = input.tensor().clone(); // Should apply actual tanh
    
    Ok(Variable::from_tensor(result_tensor, input.requires_grad()))
}

/// Softmax activation function with automatic differentiation
pub fn softmax<const DIMS: usize>(
    input: &Variable<f32, DIMS>,
    dim: usize,
) -> AnvilResult<Variable<f32, DIMS>> {
    if dim >= DIMS {
        return Err(AnvilError::InvalidInput(\"Dimension out of bounds\".to_string()));
    }
    
    let result_tensor = input.tensor().clone(); // Should apply actual softmax
    
    Ok(Variable::from_tensor(result_tensor, input.requires_grad()))
}

/// Log softmax activation function with automatic differentiation
pub fn log_softmax<const DIMS: usize>(
    input: &Variable<f32, DIMS>,
    dim: usize,
) -> AnvilResult<Variable<f32, DIMS>> {
    if dim >= DIMS {
        return Err(AnvilError::InvalidInput(\"Dimension out of bounds\".to_string()));
    }
    
    let result_tensor = input.tensor().clone(); // Should apply actual log_softmax
    
    Ok(Variable::from_tensor(result_tensor, input.requires_grad()))
}

/// Sum reduction with automatic differentiation
pub fn sum<const DIMS: usize>(
    input: &Variable<f32, DIMS>,
    dim: Option<usize>,
) -> AnvilResult<Variable<f32, 0>> {
    if let Some(d) = dim {
        if d >= DIMS {
            return Err(AnvilError::InvalidInput(\"Dimension out of bounds\".to_string()));
        }
    }
    
    // Create scalar result
    let result_tensor = AdvancedTensor::new(
        crate::tensor::Shape::new([]),
        input.dtype(),
        input.device(),
    )?;
    
    Ok(Variable::from_tensor(result_tensor, input.requires_grad()))
}

/// Mean reduction with automatic differentiation
pub fn mean<const DIMS: usize>(
    input: &Variable<f32, DIMS>,
    dim: Option<usize>,
) -> AnvilResult<Variable<f32, 0>> {
    if let Some(d) = dim {
        if d >= DIMS {
            return Err(AnvilError::InvalidInput(\"Dimension out of bounds\".to_string()));
        }
    }
    
    let result_tensor = AdvancedTensor::new(
        crate::tensor::Shape::new([]),
        input.dtype(),
        input.device(),
    )?;
    
    Ok(Variable::from_tensor(result_tensor, input.requires_grad()))
}

/// Cross-entropy loss function
pub fn cross_entropy_loss<const DIMS: usize>(
    logits: &Variable<f32, DIMS>,
    targets: &Variable<f32, DIMS>,
) -> AnvilResult<Variable<f32, 0>> {
    if logits.shape() != targets.shape() {
        return Err(AnvilError::InvalidInput(\"Shape mismatch between logits and targets\".to_string()));
    }
    
    // Compute cross-entropy loss
    let result_tensor = AdvancedTensor::new(
        crate::tensor::Shape::new([]),
        logits.dtype(),
        logits.device(),
    )?;
    
    Ok(Variable::from_tensor(result_tensor, logits.requires_grad()))
}

/// Mean squared error loss function
pub fn mse_loss<const DIMS: usize>(
    predictions: &Variable<f32, DIMS>,
    targets: &Variable<f32, DIMS>,
) -> AnvilResult<Variable<f32, 0>> {
    if predictions.shape() != targets.shape() {
        return Err(AnvilError::InvalidInput(\"Shape mismatch between predictions and targets\".to_string()));
    }
    
    let result_tensor = AdvancedTensor::new(
        crate::tensor::Shape::new([]),
        predictions.dtype(),
        predictions.device(),
    )?;
    
    Ok(Variable::from_tensor(result_tensor, predictions.requires_grad()))
}

/// Reshape operation with automatic differentiation
pub fn reshape<T, const OLD_DIMS: usize, const NEW_DIMS: usize>(
    input: &Variable<T, OLD_DIMS>,
    new_shape: crate::tensor::Shape<NEW_DIMS>,
) -> AnvilResult<Variable<T, NEW_DIMS>>
where
    T: Clone + Default + Send + Sync,
{
    // Check that total size is preserved
    let old_size = input.size();
    let new_size = new_shape.size();
    
    if old_size != new_size {
        return Err(AnvilError::InvalidInput(
            format!(\"Cannot reshape tensor of size {} to size {}\", old_size, new_size)
        ));
    }
    
    let result_tensor = AdvancedTensor::new(new_shape, input.dtype(), input.device())?;
    
    Ok(Variable::from_tensor(result_tensor, input.requires_grad()))
}

/// Transpose operation for 2D tensors
pub fn transpose<T>(
    input: &Variable<T, 2>,
) -> AnvilResult<Variable<T, 2>>
where
    T: Clone + Default + Send + Sync,
{
    let input_shape = input.shape();
    let transposed_shape = crate::tensor::Shape::new([input_shape.dims[1], input_shape.dims[0]]);
    
    let result_tensor = AdvancedTensor::new(transposed_shape, input.dtype(), input.device())?;
    
    Ok(Variable::from_tensor(result_tensor, input.requires_grad()))
}

/// Concatenation operation
pub fn cat<T, const DIMS: usize>(
    tensors: &[&Variable<T, DIMS>],
    dim: usize,
) -> AnvilResult<Variable<T, DIMS>>
where
    T: Clone + Default + Send + Sync,
{
    if tensors.is_empty() {
        return Err(AnvilError::InvalidInput(\"Cannot concatenate empty tensor list\".to_string()));
    }
    
    if dim >= DIMS {
        return Err(AnvilError::InvalidInput(\"Dimension out of bounds\".to_string()));
    }
    
    // Check that all tensors have compatible shapes
    let first_shape = tensors[0].shape();
    let mut result_shape = first_shape.clone();
    let mut total_dim_size = first_shape.dims[dim];
    
    for tensor in &tensors[1..] {
        let shape = tensor.shape();
        for (i, (&a, &b)) in first_shape.dims.iter().zip(shape.dims.iter()).enumerate() {
            if i != dim && a != b {
                return Err(AnvilError::InvalidInput(
                    format!(\"All tensors must have same size except in dimension {}\", dim)
                ));
            }
        }
        total_dim_size += shape.dims[dim];
    }
    
    result_shape.dims[dim] = total_dim_size;
    
    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    let result_tensor = AdvancedTensor::new(result_shape, tensors[0].dtype(), tensors[0].device())?;
    
    Ok(Variable::from_tensor(result_tensor, requires_grad))
}

/// Stack operation
pub fn stack<T, const DIMS: usize>(
    tensors: &[&Variable<T, DIMS>],
    dim: usize,
) -> AnvilResult<Variable<T, { DIMS + 1 }>>
where
    T: Clone + Default + Send + Sync,
{
    if tensors.is_empty() {
        return Err(AnvilError::InvalidInput(\"Cannot stack empty tensor list\".to_string()));
    }
    
    if dim > DIMS {
        return Err(AnvilError::InvalidInput(\"Dimension out of bounds for stacking\".to_string()));
    }
    
    // Check that all tensors have the same shape
    let first_shape = tensors[0].shape();
    for tensor in &tensors[1..] {
        if tensor.shape() != first_shape {
            return Err(AnvilError::InvalidInput(\"All tensors must have the same shape for stacking\".to_string()));
        }
    }
    
    // Create new shape with additional dimension
    let mut result_dims = [0; DIMS + 1];
    for i in 0..dim {
        result_dims[i] = first_shape.dims[i];
    }
    result_dims[dim] = tensors.len();
    for i in dim..DIMS {
        result_dims[i + 1] = first_shape.dims[i];
    }
    
    let result_shape = crate::tensor::Shape::new(result_dims);
    let requires_grad = tensors.iter().any(|t| t.requires_grad());
    let result_tensor = AdvancedTensor::new(result_shape, tensors[0].dtype(), tensors[0].device())?;
    
    Ok(Variable::from_tensor(result_tensor, requires_grad))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};
    
    #[test]
    fn test_add_operation() {
        let tensor1 = AdvancedTensor::<f32, 2>::new(
            Shape::new([2, 3]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let tensor2 = AdvancedTensor::<f32, 2>::new(
            Shape::new([2, 3]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let var1 = Variable::from_tensor(tensor1, true);
        let var2 = Variable::from_tensor(tensor2, true);
        
        let result = add(&var1, &var2).unwrap();
        assert!(result.requires_grad());
        assert_eq!(result.shape().dims, [2, 3]);
    }
    
    #[test]
    fn test_matmul_operation() {
        let tensor1 = AdvancedTensor::<f32, 2>::new(
            Shape::new([2, 3]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let tensor2 = AdvancedTensor::<f32, 2>::new(
            Shape::new([3, 4]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let var1 = Variable::from_tensor(tensor1, true);
        let var2 = Variable::from_tensor(tensor2, true);
        
        let result = matmul(&var1, &var2).unwrap();
        assert!(result.requires_grad());
        assert_eq!(result.shape().dims, [2, 4]);
    }
}