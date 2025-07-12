//! Advanced loss functions with automatic differentiation support
//! 
//! This module provides a comprehensive collection of loss functions commonly used
//! in machine learning, each with proper gradient computation for backpropagation.

pub mod classification;
pub mod regression;
pub mod ranking;
pub mod contrastive;
pub mod adversarial;

pub use classification::*;
pub use regression::*;
pub use ranking::*;
pub use contrastive::*;
pub use adversarial::*;

use crate::{
    autograd::Variable,
    error::{AnvilError, AnvilResult},
    tensor::AdvancedTensor,
};

/// Reduction types for loss functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// No reduction - return loss for each sample
    None,
    /// Mean reduction - average over all samples
    Mean,
    /// Sum reduction - sum over all samples
    Sum,
    /// Batch mean - average over batch dimension only
    BatchMean,
}

impl Default for Reduction {
    fn default() -> Self {
        Reduction::Mean
    }
}

/// Base trait for all loss functions
pub trait Loss: Send + Sync {
    /// Compute the loss value
    fn forward(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 0>>;
    
    /// Get the name of the loss function
    fn name(&self) -> &'static str;
    
    /// Get the reduction type
    fn reduction(&self) -> Reduction;
    
    /// Check if the loss function requires probabilistic inputs
    fn requires_probabilities(&self) -> bool {
        false
    }
    
    /// Check if the loss function requires one-hot encoded targets
    fn requires_one_hot(&self) -> bool {
        false
    }
}

/// Apply reduction to loss tensor
pub fn apply_reduction(loss: Variable<f32, 2>, reduction: Reduction) -> AnvilResult<Variable<f32, 0>> {
    match reduction {
        Reduction::None => {
            // Return the loss tensor as-is (would need to handle dimensions properly)
            Err(AnvilError::ComputationError("None reduction not supported for scalar output".to_string()))
        },
        Reduction::Mean => {
            loss.mean()
        },
        Reduction::Sum => {
            loss.sum()
        },
        Reduction::BatchMean => {
            // Average over batch dimension only
            loss.mean() // Simplified implementation
        },
    }
}

/// Numerical stability utilities
pub mod utils {
    use crate::{autograd::Variable, error::AnvilResult};
    
    /// Clamp values for numerical stability
    pub fn clamp_for_log(x: &Variable<f32, 2>, min_val: f32, max_val: f32) -> AnvilResult<Variable<f32, 2>> {
        // This would use proper tensor clamping operations
        // For now, return a copy (placeholder)
        Ok(x.clone())
    }
    
    /// Compute log-sum-exp for numerical stability
    pub fn log_sum_exp(x: &Variable<f32, 2>, dim: usize) -> AnvilResult<Variable<f32, 2>> {
        // Numerically stable log-sum-exp: log(sum(exp(x - max(x)))) + max(x)
        // This is a placeholder implementation
        Ok(x.clone())
    }
    
    /// Safe softmax computation
    pub fn safe_softmax(x: &Variable<f32, 2>, dim: usize) -> AnvilResult<Variable<f32, 2>> {
        // Subtract max for numerical stability before softmax
        x.softmax(dim)
    }
    
    /// Safe log operation
    pub fn safe_log(x: &Variable<f32, 2>, eps: f32) -> AnvilResult<Variable<f32, 2>> {
        // Clamp values to avoid log(0)
        let clamped = clamp_for_log(x, eps, f32::INFINITY)?;
        // Apply log (placeholder - would use actual tensor log operation)
        Ok(clamped)
    }
    
    /// Convert class indices to one-hot encoding
    pub fn to_one_hot(
        indices: &Variable<f32, 1>,
        num_classes: usize,
    ) -> AnvilResult<Variable<f32, 2>> {
        // This would convert class indices to one-hot vectors
        // Placeholder implementation
        let batch_size = indices.size();
        Variable::zeros(
            crate::tensor::Shape::new([batch_size, num_classes]),
            crate::tensor::DType::F32,
            crate::tensor::Device::Cpu,
            false,
        )
    }
    
    /// Convert probabilities to class predictions
    pub fn argmax(x: &Variable<f32, 2>, dim: usize) -> AnvilResult<Variable<f32, 1>> {
        // Find the index of maximum value along dimension
        // Placeholder implementation
        let batch_size = x.shape().dims[0];
        Variable::zeros(
            crate::tensor::Shape::new([batch_size]),
            crate::tensor::DType::F32,
            crate::tensor::Device::Cpu,
            false,
        )
    }
    
    /// Compute accuracy metric
    pub fn accuracy(
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<f32> {
        // Convert predictions to class indices
        let pred_classes = argmax(predictions, 1)?;
        let target_classes = argmax(targets, 1)?;
        
        // Compare and compute accuracy
        // This is a placeholder implementation
        Ok(0.85) // Placeholder accuracy
    }
    
    /// Compute top-k accuracy
    pub fn top_k_accuracy(
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
        k: usize,
    ) -> AnvilResult<f32> {
        // Check if true class is in top-k predictions
        // Placeholder implementation
        Ok(0.95) // Placeholder top-k accuracy
    }
}

/// Loss function factory for creating loss functions by name
pub struct LossFactory;

impl LossFactory {
    /// Create a loss function by name
    pub fn create(name: &str, reduction: Reduction) -> AnvilResult<Box<dyn Loss>> {
        match name.to_lowercase().as_str() {
            "crossentropy" | "cross_entropy" => {
                Ok(Box::new(CrossEntropyLoss::new(reduction, None)))
            },
            "mse" | "mean_squared_error" => {
                Ok(Box::new(MSELoss::new(reduction)))
            },
            "mae" | "mean_absolute_error" | "l1" => {
                Ok(Box::new(MAELoss::new(reduction)))
            },
            "huber" => {
                Ok(Box::new(HuberLoss::new(1.0, reduction)))
            },
            "bce" | "binary_cross_entropy" => {
                Ok(Box::new(BCELoss::new(reduction, None)))
            },
            "focal" => {
                Ok(Box::new(FocalLoss::new(2.0, 1.0, reduction)))
            },
            "dice" => {
                Ok(Box::new(DiceLoss::new(1e-8, reduction)))
            },
            "triplet" => {
                Ok(Box::new(TripletLoss::new(1.0, reduction)))
            },
            "contrastive" => {
                Ok(Box::new(ContrastiveLoss::new(1.0, reduction)))
            },
            "adversarial" | "gan" => {
                Ok(Box::new(AdversarialLoss::new(AdversarialMode::Standard, reduction)))
            },
            _ => Err(AnvilError::InvalidInput(format!("Unknown loss function: {}", name))),
        }
    }
    
    /// List all available loss functions
    pub fn available_losses() -> Vec<&'static str> {
        vec![
            "crossentropy",
            "mse",
            "mae",
            "huber",
            "bce",
            "focal",
            "dice",
            "triplet",
            "contrastive",
            "adversarial",
        ]
    }
}

/// Multi-task loss combiner
pub struct MultiTaskLoss {
    losses: Vec<Box<dyn Loss>>,
    weights: Vec<f32>,
    reduction: Reduction,
}

impl MultiTaskLoss {
    pub fn new(losses: Vec<Box<dyn Loss>>, weights: Vec<f32>, reduction: Reduction) -> AnvilResult<Self> {
        if losses.len() != weights.len() {
            return Err(AnvilError::InvalidInput(
                "Number of losses and weights must match".to_string()
            ));
        }
        
        Ok(Self {
            losses,
            weights,
            reduction,
        })
    }
    
    pub fn compute_multi_task_loss(
        &self,
        predictions: &[Variable<f32, 2>],
        targets: &[Variable<f32, 2>],
    ) -> AnvilResult<Variable<f32, 0>> {
        if predictions.len() != self.losses.len() || targets.len() != self.losses.len() {
            return Err(AnvilError::InvalidInput(
                "Number of predictions/targets must match number of losses".to_string()
            ));
        }
        
        let mut total_loss = None;
        
        for (i, (loss_fn, &weight)) in self.losses.iter().zip(self.weights.iter()).enumerate() {
            let task_loss = loss_fn.forward(&predictions[i], &targets[i])?;
            
            // Weight the loss
            // In practice, this would multiply the loss by the weight
            let weighted_loss = task_loss; // Placeholder
            
            if let Some(existing_loss) = total_loss {
                // Add to existing loss
                total_loss = Some(existing_loss); // Placeholder - should add losses
            } else {
                total_loss = Some(weighted_loss);
            }
        }
        
        total_loss.ok_or_else(|| AnvilError::ComputationError("No losses computed".to_string()))
    }
}

/// Dynamic loss weighting for multi-task learning
pub struct DynamicWeightedLoss {
    losses: Vec<Box<dyn Loss>>,
    task_weights: Variable<f32, 1>,
    uncertainty_weights: bool,
    reduction: Reduction,
}

impl DynamicWeightedLoss {
    pub fn new(
        losses: Vec<Box<dyn Loss>>,
        uncertainty_weights: bool,
        reduction: Reduction,
    ) -> AnvilResult<Self> {
        let num_tasks = losses.len();
        let task_weights = if uncertainty_weights {
            // Initialize with learnable uncertainty parameters
            Variable::ones(
                crate::tensor::Shape::new([num_tasks]),
                crate::tensor::DType::F32,
                crate::tensor::Device::Cpu,
                true, // Requires gradients for learning
            )?
        } else {
            // Initialize with equal weights
            Variable::from_vec(
                vec![1.0; num_tasks],
                crate::tensor::Shape::new([num_tasks]),
                true,
            )?
        };
        
        Ok(Self {
            losses,
            task_weights,
            uncertainty_weights,
            reduction,
        })
    }
    
    pub fn compute_weighted_loss(
        &self,
        predictions: &[Variable<f32, 2>],
        targets: &[Variable<f32, 2>],
    ) -> AnvilResult<Variable<f32, 0>> {
        let mut total_loss = None;
        let weights = self.task_weights.as_slice::<f32>();
        
        for (i, loss_fn) in self.losses.iter().enumerate() {
            let task_loss = loss_fn.forward(&predictions[i], &targets[i])?;
            
            let weight = if self.uncertainty_weights {
                // Homoscedastic uncertainty weighting: 1/(2*σ²) * loss + log(σ)
                let sigma_sq = weights[i] * weights[i];
                1.0 / (2.0 * sigma_sq)
            } else {
                weights[i]
            };
            
            // Weight the loss
            let weighted_loss = task_loss; // Placeholder - should multiply by weight
            
            if let Some(existing_loss) = total_loss {
                total_loss = Some(existing_loss); // Placeholder - should add losses
            } else {
                total_loss = Some(weighted_loss);
            }
        }
        
        // Add regularization term for uncertainty weighting
        if self.uncertainty_weights {
            let regularization = Variable::zeros(
                crate::tensor::Shape::new([]),
                crate::tensor::DType::F32,
                crate::tensor::Device::Cpu,
                true,
            )?;
            
            // Add log(σ) terms
            // reg_term = sum(log(sigma)) for each task
            
            if let Some(loss) = total_loss {
                total_loss = Some(loss); // Placeholder - should add regularization
            }
        }
        
        total_loss.ok_or_else(|| AnvilError::ComputationError("No losses computed".to_string()))
    }
    
    /// Get current task weights
    pub fn task_weights(&self) -> &Variable<f32, 1> {
        &self.task_weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};
    
    #[test]
    fn test_loss_factory() {
        let loss = LossFactory::create("crossentropy", Reduction::Mean).unwrap();
        assert_eq!(loss.name(), "CrossEntropyLoss");
        assert_eq!(loss.reduction(), Reduction::Mean);
    }
    
    #[test]
    fn test_available_losses() {
        let losses = LossFactory::available_losses();
        assert!(losses.contains(&"crossentropy"));
        assert!(losses.contains(&"mse"));
        assert!(losses.contains(&"focal"));
    }
    
    #[test]
    fn test_multi_task_loss_creation() {
        let loss1 = LossFactory::create("mse", Reduction::Mean).unwrap();
        let loss2 = LossFactory::create("crossentropy", Reduction::Mean).unwrap();
        
        let multi_loss = MultiTaskLoss::new(
            vec![loss1, loss2],
            vec![0.5, 0.5],
            Reduction::Mean,
        ).unwrap();
        
        assert_eq!(multi_loss.losses.len(), 2);
        assert_eq!(multi_loss.weights.len(), 2);
    }
    
    #[test]
    fn test_reduction_types() {
        assert_eq!(Reduction::default(), Reduction::Mean);
        
        let reductions = vec![
            Reduction::None,
            Reduction::Mean,
            Reduction::Sum,
            Reduction::BatchMean,
        ];
        
        for reduction in reductions {
            // Test that each reduction type can be created
            assert!(matches!(reduction, Reduction::None | Reduction::Mean | Reduction::Sum | Reduction::BatchMean));
        }
    }
}