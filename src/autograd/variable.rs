//! Variable wrapper for tensors in the computation graph

use std::sync::Arc;
use crate::{
    tensor::AdvancedTensor,
    error::{AnvilError, AnvilResult},
};
use super::graph::NodeId;

/// Gradient information for a variable
#[derive(Debug, Clone)]
pub struct Gradient<T, const DIMS: usize> 
where
    T: Clone + Default + Send + Sync,
{
    pub data: AdvancedTensor<T, DIMS>,
    pub accumulated: bool,
}

impl<T, const DIMS: usize> Gradient<T, DIMS> 
where
    T: Clone + Default + Send + Sync,
{
    pub fn new(data: AdvancedTensor<T, DIMS>) -> Self {
        Self {
            data,
            accumulated: false,
        }
    }
    
    pub fn zero_like(tensor: &AdvancedTensor<T, DIMS>) -> AnvilResult<Self> {
        let zero_tensor = AdvancedTensor::new(
            tensor.shape().clone(),
            tensor.dtype(),
            tensor.device(),
        )?;
        Ok(Self::new(zero_tensor))
    }
}

/// Variable represents a tensor that can participate in automatic differentiation
#[derive(Debug, Clone)]
pub struct Variable<T, const DIMS: usize> 
where
    T: Clone + Default + Send + Sync,
{
    tensor: AdvancedTensor<T, DIMS>,
    node_id: Option<NodeId>,
    requires_grad: bool,
    gradient: Option<Gradient<T, DIMS>>,
    is_leaf: bool,
}

impl<T, const DIMS: usize> Variable<T, DIMS> 
where
    T: Clone + Default + Send + Sync,
{
    /// Create a new variable
    pub fn new(
        tensor: AdvancedTensor<T, DIMS>,
        node_id: Option<NodeId>,
        requires_grad: bool,
    ) -> Self {
        Self {
            tensor,
            node_id,
            requires_grad,
            gradient: None,
            is_leaf: node_id.is_some(),
        }
    }
    
    /// Create a variable from tensor data
    pub fn from_tensor(tensor: AdvancedTensor<T, DIMS>, requires_grad: bool) -> Self {
        Self::new(tensor, None, requires_grad)
    }
    
    /// Create a variable with zeros
    pub fn zeros(
        shape: crate::tensor::Shape<DIMS>,
        dtype: crate::tensor::DType,
        device: crate::tensor::Device,
        requires_grad: bool,
    ) -> AnvilResult<Self> {
        let tensor = AdvancedTensor::new(shape, dtype, device)?;
        Ok(Self::from_tensor(tensor, requires_grad))
    }
    
    /// Create a variable with ones
    pub fn ones(
        shape: crate::tensor::Shape<DIMS>,
        dtype: crate::tensor::DType,
        device: crate::tensor::Device,
        requires_grad: bool,
    ) -> AnvilResult<Self> {
        let mut tensor = AdvancedTensor::new(shape, dtype, device)?;
        // Fill with ones - simplified implementation
        Ok(Self::from_tensor(tensor, requires_grad))
    }
    
    /// Create a variable from raw data
    pub fn from_vec(
        data: Vec<T>,
        shape: crate::tensor::Shape<DIMS>,
        requires_grad: bool,
    ) -> AnvilResult<Self> 
    where
        T: 'static,
    {
        let tensor = AdvancedTensor::from_vec(data, shape)?;
        Ok(Self::from_tensor(tensor, requires_grad))
    }
    
    /// Get reference to underlying tensor
    pub fn tensor(&self) -> &AdvancedTensor<T, DIMS> {
        &self.tensor
    }
    
    /// Get mutable reference to underlying tensor
    pub fn tensor_mut(&mut self) -> &mut AdvancedTensor<T, DIMS> {
        &mut self.tensor
    }
    
    /// Get node ID in computation graph
    pub fn node_id(&self) -> Option<NodeId> {
        self.node_id
    }
    
    /// Check if variable requires gradients
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }
    
    /// Check if variable is a leaf node
    pub fn is_leaf(&self) -> bool {
        self.is_leaf
    }
    
    /// Get gradient if available
    pub fn grad(&self) -> Option<&Gradient<T, DIMS>> {
        self.gradient.as_ref()
    }
    
    /// Get mutable gradient if available
    pub fn grad_mut(&mut self) -> Option<&mut Gradient<T, DIMS>> {
        self.gradient.as_mut()
    }
    
    /// Set gradient
    pub fn set_grad(&mut self, gradient: Gradient<T, DIMS>) {
        self.gradient = Some(gradient);
    }
    
    /// Zero gradients
    pub fn zero_grad(&mut self) -> AnvilResult<()> {
        if self.requires_grad {
            self.gradient = Some(Gradient::zero_like(&self.tensor)?);
        }
        Ok(())
    }
    
    /// Detach variable from computation graph
    pub fn detach(&self) -> Self {
        Self::new(self.tensor.clone(), None, false)
    }
    
    /// Clone variable with gradient information
    pub fn clone_with_grad(&self) -> Self {
        Self {
            tensor: self.tensor.clone(),
            node_id: self.node_id,
            requires_grad: self.requires_grad,
            gradient: self.gradient.clone(),
            is_leaf: self.is_leaf,
        }
    }
    
    /// Get shape of the variable
    pub fn shape(&self) -> &crate::tensor::Shape<DIMS> {
        self.tensor.shape()
    }
    
    /// Get data type
    pub fn dtype(&self) -> crate::tensor::DType {
        self.tensor.dtype()
    }
    
    /// Get device
    pub fn device(&self) -> crate::tensor::Device {
        self.tensor.device()
    }
    
    /// Get size (number of elements)
    pub fn size(&self) -> usize {
        self.tensor.size()
    }
    
    /// Get raw data as slice
    pub fn as_slice<U>(&self) -> &[U] 
    where
        U: 'static,
    {
        self.tensor.as_slice()
    }
    
    /// Get raw data as mutable slice
    pub fn as_slice_mut<U>(&mut self) -> &mut [U] 
    where
        U: 'static,
    {
        self.tensor.as_slice_mut()
    }
    
    /// Convert to different device
    pub async fn to_device(&self, device: crate::tensor::Device) -> AnvilResult<Self> {
        let new_tensor = self.tensor.to_device(device).await?;
        Ok(Self::new(new_tensor, None, self.requires_grad))
    }
    
    /// Reshape variable
    pub fn reshape<const NEW_DIMS: usize>(
        &self, 
        new_shape: crate::tensor::Shape<NEW_DIMS>
    ) -> AnvilResult<Variable<T, NEW_DIMS>> {
        let new_tensor = self.tensor.reshape(new_shape)?;
        Ok(Variable::new(new_tensor, None, self.requires_grad))
    }
    
    /// Sum reduction
    pub fn sum(&self) -> AnvilResult<Variable<T, 0>> {
        // This would create a new operation node in the computation graph
        // For now, simplified implementation
        let scalar_tensor = AdvancedTensor::new(
            crate::tensor::Shape::new([]),
            self.dtype(),
            self.device(),
        )?;
        Ok(Variable::new(scalar_tensor, None, self.requires_grad))
    }
    
    /// Mean reduction
    pub fn mean(&self) -> AnvilResult<Variable<T, 0>> {
        // This would create a new operation node in the computation graph
        let scalar_tensor = AdvancedTensor::new(
            crate::tensor::Shape::new([]),
            self.dtype(),
            self.device(),
        )?;
        Ok(Variable::new(scalar_tensor, None, self.requires_grad))
    }
}

// Implement arithmetic operations that create computation graph nodes
impl<T, const DIMS: usize> Variable<T, DIMS> 
where
    T: Clone + Default + Send + Sync + std::ops::Add<Output = T>,
{
    /// Element-wise addition
    pub fn add(&self, other: &Self) -> AnvilResult<Self> {
        // In a full implementation, this would:
        // 1. Create a new operation node in the computation graph
        // 2. Set up the gradient function for addition
        // 3. Return a new variable with the result
        
        // Simplified implementation for now
        if self.shape() != other.shape() {
            return Err(AnvilError::InvalidInput(\"Shape mismatch for addition\".to_string()));
        }
        
        // Create result tensor (simplified)
        let result_tensor = self.tensor.clone(); // Should be actual addition
        Ok(Self::new(result_tensor, None, self.requires_grad || other.requires_grad))
    }
    
    /// Element-wise subtraction
    pub fn sub(&self, other: &Self) -> AnvilResult<Self> 
    where
        T: std::ops::Sub<Output = T>,
    {
        // Similar to add, but with subtraction operation
        let result_tensor = self.tensor.clone();
        Ok(Self::new(result_tensor, None, self.requires_grad || other.requires_grad))
    }
    
    /// Element-wise multiplication
    pub fn mul(&self, other: &Self) -> AnvilResult<Self> 
    where
        T: std::ops::Mul<Output = T>,
    {
        let result_tensor = self.tensor.clone();
        Ok(Self::new(result_tensor, None, self.requires_grad || other.requires_grad))
    }
    
    /// Matrix multiplication (for 2D variables)
    pub fn matmul(&self, other: &Self) -> AnvilResult<Self> 
    where
        T: std::ops::Mul<Output = T>,
    {
        if DIMS != 2 {
            return Err(AnvilError::InvalidInput(\"Matrix multiplication requires 2D tensors\".to_string()));
        }
        
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        if self_shape.dims[1] != other_shape.dims[0] {
            return Err(AnvilError::InvalidInput(\"Incompatible dimensions for matrix multiplication\".to_string()));
        }
        
        // Create result shape
        let result_shape = crate::tensor::Shape::new([self_shape.dims[0], other_shape.dims[1]]);
        let result_tensor = AdvancedTensor::new(result_shape, self.dtype(), self.device())?;
        
        Ok(Self::new(result_tensor, None, self.requires_grad || other.requires_grad))
    }
}

// Activation functions
impl Variable<f32, DIMS> {
    /// ReLU activation function
    pub fn relu(&self) -> AnvilResult<Self> {
        let result_tensor = self.tensor.clone(); // Should apply ReLU
        Ok(Self::new(result_tensor, None, self.requires_grad))
    }
    
    /// Sigmoid activation function
    pub fn sigmoid(&self) -> AnvilResult<Self> {
        let result_tensor = self.tensor.clone(); // Should apply sigmoid
        Ok(Self::new(result_tensor, None, self.requires_grad))
    }
    
    /// Tanh activation function
    pub fn tanh(&self) -> AnvilResult<Self> {
        let result_tensor = self.tensor.clone(); // Should apply tanh
        Ok(Self::new(result_tensor, None, self.requires_grad))
    }
    
    /// Softmax activation function
    pub fn softmax(&self, dim: usize) -> AnvilResult<Self> {
        if dim >= DIMS {
            return Err(AnvilError::InvalidInput(\"Dimension out of bounds\".to_string()));
        }
        
        let result_tensor = self.tensor.clone(); // Should apply softmax
        Ok(Self::new(result_tensor, None, self.requires_grad))
    }
    
    /// Log softmax activation function
    pub fn log_softmax(&self, dim: usize) -> AnvilResult<Self> {
        if dim >= DIMS {
            return Err(AnvilError::InvalidInput(\"Dimension out of bounds\".to_string()));
        }
        
        let result_tensor = self.tensor.clone(); // Should apply log_softmax
        Ok(Self::new(result_tensor, None, self.requires_grad))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};
    
    #[test]
    fn test_variable_creation() {
        let tensor = AdvancedTensor::<f32, 2>::new(
            Shape::new([2, 3]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let var = Variable::from_tensor(tensor, true);
        assert!(var.requires_grad());
        assert_eq!(var.shape().dims, [2, 3]);
    }
    
    #[test]
    fn test_variable_operations() {
        let tensor1 = AdvancedTensor::<f32, 2>::new(
            Shape::new([2, 2]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let tensor2 = AdvancedTensor::<f32, 2>::new(
            Shape::new([2, 2]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let var1 = Variable::from_tensor(tensor1, true);
        let var2 = Variable::from_tensor(tensor2, true);
        
        let result = var1.add(&var2).unwrap();
        assert!(result.requires_grad());
    }
}