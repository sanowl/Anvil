//! Automatic differentiation system with computation graph
//! 
//! This module implements reverse-mode automatic differentiation (backpropagation)
//! with a dynamic computation graph for flexible model definition and training.

pub mod graph;
pub mod variable;
pub mod function;
pub mod ops;

pub use graph::{ComputationGraph, GraphNode, NodeId};
pub use variable::{Variable, Gradient};
pub use function::{Function, FunctionContext};
pub use ops::*;

use std::sync::Arc;
use crate::{
    tensor::AdvancedTensor,
    error::{AnvilError, AnvilResult},
};

/// Main autograd engine that manages computation graphs and gradient computation
pub struct AutogradEngine {
    graph: ComputationGraph,
    retain_graph: bool,
}

impl AutogradEngine {
    pub fn new() -> Self {
        Self {
            graph: ComputationGraph::new(),
            retain_graph: false,
        }
    }
    
    /// Enable graph retention for multiple backward passes
    pub fn retain_graph(mut self) -> Self {
        self.retain_graph = true;
        self
    }
    
    /// Create a variable (leaf node) that requires gradients
    pub fn variable<T, const DIMS: usize>(
        &mut self, 
        tensor: AdvancedTensor<T, DIMS>,
        requires_grad: bool
    ) -> Variable<T, DIMS> 
    where
        T: Clone + Default + Send + Sync,
    {
        let node_id = self.graph.create_leaf_node(tensor.clone(), requires_grad);
        Variable::new(tensor, Some(node_id), requires_grad)
    }
    
    /// Run backward pass from a scalar loss
    pub fn backward<T>(&mut self, loss: &Variable<T, 0>) -> AnvilResult<()> 
    where
        T: Clone + Default + Send + Sync + std::fmt::Debug,
    {
        if let Some(node_id) = loss.node_id() {
            self.graph.backward(node_id)?;
            
            if !self.retain_graph {
                self.graph.clear();
            }
        }
        Ok(())
    }
    
    /// Get accumulated gradients for a variable
    pub fn gradients<T, const DIMS: usize>(&self, var: &Variable<T, DIMS>) -> Option<&AdvancedTensor<T, DIMS>> 
    where
        T: Clone + Default + Send + Sync,
    {
        if let Some(node_id) = var.node_id() {
            self.graph.get_gradient(node_id)
        } else {
            None
        }
    }
    
    /// Zero all gradients in the computation graph
    pub fn zero_grad(&mut self) {
        self.graph.zero_gradients();
    }
    
    /// Clear the computation graph
    pub fn clear_graph(&mut self) {
        self.graph.clear();
    }
}