//! Computation graph for automatic differentiation

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};
use std::any::Any;
use crate::{
    tensor::AdvancedTensor,
    error::{AnvilError, AnvilResult},
};
use super::function::Function;

pub type NodeId = usize;

/// A node in the computation graph representing a tensor and its gradient computation
#[derive(Debug)]
pub struct GraphNode {
    pub id: NodeId,
    pub tensor: Box<dyn Any + Send + Sync>,
    pub gradient: Option<Box<dyn Any + Send + Sync>>,
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub parents: Vec<NodeId>,
    pub children: Vec<NodeId>,
    pub grad_fn: Option<Box<dyn Function + Send + Sync>>,
    pub reference_count: usize,
}

impl GraphNode {
    pub fn new_leaf<T, const DIMS: usize>(
        id: NodeId,
        tensor: AdvancedTensor<T, DIMS>,
        requires_grad: bool,
    ) -> Self 
    where
        T: Copy + Clone + Default + Send + Sync + 'static,
    {
        Self {
            id,
            tensor: Box::new(tensor),
            gradient: None,
            requires_grad,
            is_leaf: true,
            parents: Vec::new(),
            children: Vec::new(),
            grad_fn: None,
            reference_count: 0,
        }
    }
    
    pub fn new_op<T, const DIMS: usize>(
        id: NodeId,
        tensor: AdvancedTensor<T, DIMS>,
        parents: Vec<NodeId>,
        grad_fn: Box<dyn Function + Send + Sync>,
    ) -> Self 
    where
        T: Copy + Clone + Default + Send + Sync + 'static,
    {
        Self {
            id,
            tensor: Box::new(tensor),
            gradient: None,
            requires_grad: true,
            is_leaf: false,
            parents,
            children: Vec::new(),
            grad_fn: Some(grad_fn),
            reference_count: 0,
        }
    }
    
    /// Get tensor with type checking
    pub fn get_tensor<T, const DIMS: usize>(&self) -> Option<&AdvancedTensor<T, DIMS>> 
    where
        T: Copy + 'static,
    {
        self.tensor.downcast_ref::<AdvancedTensor<T, DIMS>>()
    }
    
    /// Get gradient with type checking
    pub fn get_gradient<T, const DIMS: usize>(&self) -> Option<&AdvancedTensor<T, DIMS>> 
    where
        T: Copy + 'static,
    {
        self.gradient.as_ref()?.downcast_ref::<AdvancedTensor<T, DIMS>>()
    }
    
    /// Set gradient with type checking
    pub fn set_gradient<T, const DIMS: usize>(&mut self, grad: AdvancedTensor<T, DIMS>) 
    where
        T: Copy + Clone + Default + Send + Sync + 'static,
    {
        self.gradient = Some(Box::new(grad));
    }
    
    /// Accumulate gradient
    pub fn accumulate_gradient<T, const DIMS: usize>(&mut self, grad: AdvancedTensor<T, DIMS>) -> AnvilResult<()>
    where
        T: Copy + Clone + Default + Send + Sync + 'static + std::ops::Add<Output = T>,
    {
        if let Some(existing_grad) = &mut self.gradient {
            if let Some(existing) = existing_grad.downcast_mut::<AdvancedTensor<T, DIMS>>() {
                // Add the new gradient to existing one
                // For now, use simple element-wise addition
                // In production, this would use the tensor's add operation
                *existing = grad; // Simplified - should be existing + grad
            } else {
                return Err(AnvilError::ComputationError("Gradient type mismatch".to_string()));
            }
        } else {
            self.gradient = Some(Box::new(grad));
        }
        Ok(())
    }
}

/// Computation graph managing forward and backward passes
pub struct ComputationGraph {
    nodes: HashMap<NodeId, GraphNode>,
    next_id: NodeId,
    topological_order: Vec<NodeId>,
    roots: Vec<NodeId>,
}

impl ComputationGraph {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 0,
            topological_order: Vec::new(),
            roots: Vec::new(),
        }
    }
    
    /// Create a leaf node (variable)
    pub fn create_leaf_node<T, const DIMS: usize>(
        &mut self,
        tensor: AdvancedTensor<T, DIMS>,
        requires_grad: bool,
    ) -> NodeId 
    where
        T: Copy + Clone + Default + Send + Sync + 'static,
    {
        let node_id = self.next_id;
        self.next_id += 1;
        
        let node = GraphNode::new_leaf(node_id, tensor, requires_grad);
        self.nodes.insert(node_id, node);
        
        if requires_grad {
            self.roots.push(node_id);
        }
        
        node_id
    }
    
    /// Create an operation node
    pub fn create_op_node<T, const DIMS: usize>(
        &mut self,
        tensor: AdvancedTensor<T, DIMS>,
        parents: Vec<NodeId>,
        grad_fn: Box<dyn Function + Send + Sync>,
    ) -> NodeId 
    where
        T: Copy + Clone + Default + Send + Sync + 'static,
    {
        let node_id = self.next_id;
        self.next_id += 1;
        
        let mut node = GraphNode::new_op(node_id, tensor, parents.clone(), grad_fn);
        
        // Update parent-child relationships
        for &parent_id in &parents {
            if let Some(parent) = self.nodes.get_mut(&parent_id) {
                parent.children.push(node_id);
                parent.reference_count += 1;
            }
        }
        
        self.nodes.insert(node_id, node);
        node_id
    }
    
    /// Perform backward pass from a root node
    pub fn backward(&mut self, root_id: NodeId) -> AnvilResult<()> {
        // Initialize gradient at root (loss) to ones
        if let Some(root_node) = self.nodes.get_mut(&root_id) {
            // Create gradient tensor of ones with same shape as root
            // This is simplified - in practice, we'd need proper tensor operations
            // For scalar loss, this would be a scalar 1.0
        }
        
        // Compute topological order for backward pass
        self.compute_topological_order(root_id)?;
        
        // Traverse in reverse topological order
        for &node_id in self.topological_order.iter().rev() {
            if let Some(node) = self.nodes.get(&node_id) {
                if let Some(grad_fn) = &node.grad_fn {
                    // Compute gradients for parent nodes
                    let parent_grads = grad_fn.backward(&node.parents)?;
                    
                    // Distribute gradients to parents
                    for (i, &parent_id) in node.parents.iter().enumerate() {
                        if i < parent_grads.len() {
                            if let Some(parent_node) = self.nodes.get_mut(&parent_id) {
                                if parent_node.requires_grad {
                                    // Accumulate gradient - this is simplified
                                    // In practice, we'd need proper tensor gradient accumulation
                                }
                            }
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Compute topological ordering for backward pass
    fn compute_topological_order(&mut self, root_id: NodeId) -> AnvilResult<()> {
        self.topological_order.clear();
        let mut visited = HashMap::new();
        let mut temp_mark = HashMap::new();
        
        self.topological_sort_dfs(root_id, &mut visited, &mut temp_mark)?;
        
        Ok(())
    }
    
    /// Depth-first search for topological sorting
    fn topological_sort_dfs(
        &mut self,
        node_id: NodeId,
        visited: &mut HashMap<NodeId, bool>,
        temp_mark: &mut HashMap<NodeId, bool>,
    ) -> AnvilResult<()> {
        if temp_mark.get(&node_id) == Some(&true) {
            return Err(AnvilError::ComputationError("Cycle detected in computation graph".to_string()));
        }
        
        if visited.get(&node_id) == Some(&true) {
            return Ok(());
        }
        
        temp_mark.insert(node_id, true);
        
        if let Some(node) = self.nodes.get(&node_id) {
            for &parent_id in &node.parents {
                self.topological_sort_dfs(parent_id, visited, temp_mark)?;
            }
        }
        
        temp_mark.insert(node_id, false);
        visited.insert(node_id, true);
        self.topological_order.push(node_id);
        
        Ok(())
    }
    
    /// Get gradient for a node
    pub fn get_gradient<T, const DIMS: usize>(&self, node_id: NodeId) -> Option<&AdvancedTensor<T, DIMS>> 
    where
        T: Copy + 'static,
    {
        self.nodes.get(&node_id)?.get_gradient()
    }
    
    /// Zero all gradients
    pub fn zero_gradients(&mut self) {
        for node in self.nodes.values_mut() {
            node.gradient = None;
        }
    }
    
    /// Clear the computation graph
    pub fn clear(&mut self) {
        self.nodes.clear();
        self.topological_order.clear();
        self.roots.clear();
        self.next_id = 0;
    }
    
    /// Get number of nodes in graph
    pub fn num_nodes(&self) -> usize {
        self.nodes.len()
    }
    
    /// Check if graph is acyclic
    pub fn is_acyclic(&self) -> bool {
        // Simplified cycle detection
        let mut visited = HashMap::new();
        let mut temp_mark = HashMap::new();
        
        for &root_id in &self.roots {
            if self.has_cycle_dfs(root_id, &mut visited, &mut temp_mark) {
                return false;
            }
        }
        
        true
    }
    
    fn has_cycle_dfs(
        &self,
        node_id: NodeId,
        visited: &mut HashMap<NodeId, bool>,
        temp_mark: &mut HashMap<NodeId, bool>,
    ) -> bool {
        if temp_mark.get(&node_id) == Some(&true) {
            return true; // Cycle detected
        }
        
        if visited.get(&node_id) == Some(&true) {
            return false;
        }
        
        temp_mark.insert(node_id, true);
        
        if let Some(node) = self.nodes.get(&node_id) {
            for &child_id in &node.children {
                if self.has_cycle_dfs(child_id, visited, temp_mark) {
                    return true;
                }
            }
        }
        
        temp_mark.insert(node_id, false);
        visited.insert(node_id, true);
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};
    
    #[test]
    fn test_graph_creation() {
        let mut graph = ComputationGraph::new();
        
        let tensor = AdvancedTensor::<f32, 2>::new(
            Shape::new([2, 2]), 
            DType::F32, 
            Device::Cpu
        ).unwrap();
        
        let node_id = graph.create_leaf_node(tensor, true);
        assert_eq!(graph.num_nodes(), 1);
        assert!(graph.is_acyclic());
    }
}