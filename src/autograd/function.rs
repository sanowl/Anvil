//! Function trait and context for automatic differentiation

use std::any::Any;
use std::sync::Arc;
use crate::{
    tensor::AdvancedTensor,
    error::{AnvilError, AnvilResult},
};
use super::graph::NodeId;

/// Context passed to functions during forward and backward passes
#[derive(Debug, Clone)]
pub struct FunctionContext {
    pub saved_tensors: Vec<Box<dyn Any + Send + Sync>>,
    pub needs_input_grad: Vec<bool>,
    pub input_shapes: Vec<Vec<usize>>,
}

impl FunctionContext {
    pub fn new() -> Self {
        Self {
            saved_tensors: Vec::new(),
            needs_input_grad: Vec::new(),
            input_shapes: Vec::new(),
        }
    }
    
    /// Save tensor for backward pass
    pub fn save_for_backward<T, const DIMS: usize>(&mut self, tensor: &AdvancedTensor<T, DIMS>)
    where
        T: Clone + Send + Sync + 'static,
    {
        self.saved_tensors.push(Box::new(tensor.clone()));
    }
    
    /// Get saved tensor with type checking
    pub fn get_saved_tensor<T, const DIMS: usize>(&self, index: usize) -> Option<&AdvancedTensor<T, DIMS>>
    where
        T: 'static,
    {
        self.saved_tensors.get(index)?.downcast_ref()
    }
    
    /// Mark input as needing gradients
    pub fn mark_needs_input_grad(&mut self, index: usize, needs_grad: bool) {
        if self.needs_input_grad.len() <= index {
            self.needs_input_grad.resize(index + 1, false);
        }
        self.needs_input_grad[index] = needs_grad;
    }
    
    /// Check if input needs gradients
    pub fn needs_input_grad(&self, index: usize) -> bool {
        self.needs_input_grad.get(index).copied().unwrap_or(false)
    }
    
    /// Save input shape
    pub fn save_input_shape(&mut self, shape: &[usize]) {
        self.input_shapes.push(shape.to_vec());
    }
    
    /// Get saved input shape
    pub fn get_input_shape(&self, index: usize) -> Option<&[usize]> {
        self.input_shapes.get(index).map(|v| v.as_slice())
    }
}

/// Trait for functions that can be differentiated
pub trait Function {
    /// Forward pass - compute output from inputs
    fn forward(&self, inputs: &[NodeId], ctx: &mut FunctionContext) -> AnvilResult<Box<dyn Any + Send + Sync>>;
    
    /// Backward pass - compute gradients with respect to inputs
    fn backward(&self, inputs: &[NodeId]) -> AnvilResult<Vec<Box<dyn Any + Send + Sync>>>;
    
    /// Get function name for debugging
    fn name(&self) -> &'static str;
    
    /// Check if function is differentiable
    fn is_differentiable(&self) -> bool {
        true
    }
}

/// Addition function for automatic differentiation
#[derive(Debug, Clone)]
pub struct AddFunction {
    context: FunctionContext,
}

impl AddFunction {
    pub fn new() -> Self {
        Self {
            context: FunctionContext::new(),
        }
    }
}

impl Function for AddFunction {
    fn forward(&self, inputs: &[NodeId], ctx: &mut FunctionContext) -> AnvilResult<Box<dyn Any + Send + Sync>> {
        if inputs.len() != 2 {
            return Err(AnvilError::InvalidInput("Add function requires exactly 2 inputs".to_string()));
        }
        
        // Mark both inputs as needing gradients
        ctx.mark_needs_input_grad(0, true);
        ctx.mark_needs_input_grad(1, true);
        
        // For now, return a placeholder
        // In full implementation, this would perform actual tensor addition
        Err(AnvilError::ComputationError("Add function not fully implemented".to_string()))
    }
    
    fn backward(&self, inputs: &[NodeId]) -> AnvilResult<Vec<Box<dyn Any + Send + Sync>>> {
        // For addition: d/dx (x + y) = 1, d/dy (x + y) = 1
        // So gradients are just passed through unchanged
        
        // Return gradients for both inputs
        // In full implementation, these would be the actual gradient tensors
        Ok(vec![])
    }
    
    fn name(&self) -> &'static str {
        "AddFunction"
    }
}

/// Multiplication function for automatic differentiation
#[derive(Debug, Clone)]
pub struct MulFunction {
    context: FunctionContext,
}

impl MulFunction {
    pub fn new() -> Self {
        Self {
            context: FunctionContext::new(),
        }
    }
}

impl Function for MulFunction {
    fn forward(&self, inputs: &[NodeId], ctx: &mut FunctionContext) -> AnvilResult<Box<dyn Any + Send + Sync>> {
        if inputs.len() != 2 {
            return Err(AnvilError::InvalidInput("Mul function requires exactly 2 inputs".to_string()));
        }
        
        // Save inputs for backward pass (needed for product rule)
        ctx.mark_needs_input_grad(0, true);
        ctx.mark_needs_input_grad(1, true);
        
        // In full implementation, would save the input tensors for backward pass
        // ctx.save_for_backward(&input1);
        // ctx.save_for_backward(&input2);
        
        Err(AnvilError::ComputationError("Mul function not fully implemented".to_string()))
    }
    
    fn backward(&self, inputs: &[NodeId]) -> AnvilResult<Vec<Box<dyn Any + Send + Sync>>> {
        // For multiplication: d/dx (x * y) = y, d/dy (x * y) = x
        // Need to retrieve saved tensors and multiply gradient by the other input
        
        Ok(vec![])
    }
    
    fn name(&self) -> &'static str {
        "MulFunction"
    }
}

/// Matrix multiplication function
#[derive(Debug, Clone)]
pub struct MatMulFunction {
    context: FunctionContext,
}

impl MatMulFunction {
    pub fn new() -> Self {
        Self {
            context: FunctionContext::new(),
        }
    }
}

impl Function for MatMulFunction {
    fn forward(&self, inputs: &[NodeId], ctx: &mut FunctionContext) -> AnvilResult<Box<dyn Any + Send + Sync>> {
        if inputs.len() != 2 {
            return Err(AnvilError::InvalidInput("MatMul function requires exactly 2 inputs".to_string()));
        }
        
        ctx.mark_needs_input_grad(0, true);
        ctx.mark_needs_input_grad(1, true);
        
        // Save inputs for backward pass
        // For matrix multiplication, we need both input matrices to compute gradients
        
        Err(AnvilError::ComputationError("MatMul function not fully implemented".to_string()))
    }
    
    fn backward(&self, inputs: &[NodeId]) -> AnvilResult<Vec<Box<dyn Any + Send + Sync>>> {
        // For matrix multiplication C = A @ B:
        // dL/dA = dL/dC @ B^T
        // dL/dB = A^T @ dL/dC
        
        Ok(vec![])
    }
    
    fn name(&self) -> &'static str {
        "MatMulFunction"
    }
}

/// ReLU activation function
#[derive(Debug, Clone)]
pub struct ReLUFunction {
    context: FunctionContext,
}

impl ReLUFunction {
    pub fn new() -> Self {
        Self {
            context: FunctionContext::new(),
        }
    }
}

impl Function for ReLUFunction {
    fn forward(&self, inputs: &[NodeId], ctx: &mut FunctionContext) -> AnvilResult<Box<dyn Any + Send + Sync>> {
        if inputs.len() != 1 {
            return Err(AnvilError::InvalidInput("ReLU function requires exactly 1 input".to_string()));
        }
        
        ctx.mark_needs_input_grad(0, true);
        
        // Save input for backward pass (needed to determine where gradient should be zero)
        
        Err(AnvilError::ComputationError("ReLU function not fully implemented".to_string()))
    }
    
    fn backward(&self, inputs: &[NodeId]) -> AnvilResult<Vec<Box<dyn Any + Send + Sync>>> {
        // For ReLU: d/dx max(0, x) = 1 if x > 0, 0 otherwise
        // Need to create a mask based on the saved input
        
        Ok(vec![])
    }
    
    fn name(&self) -> &'static str {
        "ReLUFunction"
    }
}

/// Sigmoid activation function
#[derive(Debug, Clone)]
pub struct SigmoidFunction {
    context: FunctionContext,
}

impl SigmoidFunction {
    pub fn new() -> Self {
        Self {
            context: FunctionContext::new(),
        }
    }
}

impl Function for SigmoidFunction {
    fn forward(&self, inputs: &[NodeId], ctx: &mut FunctionContext) -> AnvilResult<Box<dyn Any + Send + Sync>> {
        if inputs.len() != 1 {
            return Err(AnvilError::InvalidInput("Sigmoid function requires exactly 1 input".to_string()));
        }
        
        ctx.mark_needs_input_grad(0, true);
        
        // Save output for backward pass (sigmoid derivative can be computed from output)
        
        Err(AnvilError::ComputationError("Sigmoid function not fully implemented".to_string()))
    }
    
    fn backward(&self, inputs: &[NodeId]) -> AnvilResult<Vec<Box<dyn Any + Send + Sync>>> {
        // For sigmoid: d/dx σ(x) = σ(x) * (1 - σ(x))
        // Can compute from saved output
        
        Ok(vec![])
    }
    
    fn name(&self) -> &'static str {
        "SigmoidFunction"
    }
}

/// Softmax function
#[derive(Debug, Clone)]
pub struct SoftmaxFunction {
    dim: usize,
    context: FunctionContext,
}

impl SoftmaxFunction {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            context: FunctionContext::new(),
        }
    }
}

impl Function for SoftmaxFunction {
    fn forward(&self, inputs: &[NodeId], ctx: &mut FunctionContext) -> AnvilResult<Box<dyn Any + Send + Sync>> {
        if inputs.len() != 1 {
            return Err(AnvilError::InvalidInput("Softmax function requires exactly 1 input".to_string()));
        }
        
        ctx.mark_needs_input_grad(0, true);
        
        // Save output for backward pass
        
        Err(AnvilError::ComputationError("Softmax function not fully implemented".to_string()))
    }
    
    fn backward(&self, inputs: &[NodeId]) -> AnvilResult<Vec<Box<dyn Any + Send + Sync>>> {
        // Softmax derivative is more complex: 
        // d/dx_i softmax(x)_j = softmax(x)_j * (δ_ij - softmax(x)_i)
        // where δ_ij is the Kronecker delta
        
        Ok(vec![])
    }
    
    fn name(&self) -> &'static str {
        "SoftmaxFunction"
    }
}

/// Cross-entropy loss function
#[derive(Debug, Clone)]
pub struct CrossEntropyFunction {
    context: FunctionContext,
}

impl CrossEntropyFunction {
    pub fn new() -> Self {
        Self {
            context: FunctionContext::new(),
        }
    }
}

impl Function for CrossEntropyFunction {
    fn forward(&self, inputs: &[NodeId], ctx: &mut FunctionContext) -> AnvilResult<Box<dyn Any + Send + Sync>> {
        if inputs.len() != 2 {
            return Err(AnvilError::InvalidInput("CrossEntropy function requires exactly 2 inputs (logits, targets)".to_string()));
        }
        
        ctx.mark_needs_input_grad(0, true);  // logits need gradients
        ctx.mark_needs_input_grad(1, false); // targets typically don't need gradients
        
        // Save inputs for backward pass
        
        Err(AnvilError::ComputationError("CrossEntropy function not fully implemented".to_string()))
    }
    
    fn backward(&self, inputs: &[NodeId]) -> AnvilResult<Vec<Box<dyn Any + Send + Sync>>> {
        // Cross-entropy gradient: softmax(logits) - targets
        
        Ok(vec![])
    }
    
    fn name(&self) -> &'static str {
        "CrossEntropyFunction"
    }
}

/// Mean squared error loss function
#[derive(Debug, Clone)]
pub struct MSEFunction {
    context: FunctionContext,
}

impl MSEFunction {
    pub fn new() -> Self {
        Self {
            context: FunctionContext::new(),
        }
    }
}

impl Function for MSEFunction {
    fn forward(&self, inputs: &[NodeId], ctx: &mut FunctionContext) -> AnvilResult<Box<dyn Any + Send + Sync>> {
        if inputs.len() != 2 {
            return Err(AnvilError::InvalidInput("MSE function requires exactly 2 inputs (predictions, targets)".to_string()));
        }
        
        ctx.mark_needs_input_grad(0, true);  // predictions need gradients
        ctx.mark_needs_input_grad(1, false); // targets typically don't need gradients
        
        Err(AnvilError::ComputationError("MSE function not fully implemented".to_string()))
    }
    
    fn backward(&self, inputs: &[NodeId]) -> AnvilResult<Vec<Box<dyn Any + Send + Sync>>> {
        // MSE gradient: 2 * (predictions - targets) / N
        
        Ok(vec![])
    }
    
    fn name(&self) -> &'static str {
        "MSEFunction"
    }
}