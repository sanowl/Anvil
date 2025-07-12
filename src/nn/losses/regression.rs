//! Regression loss functions

use crate::{autograd::Variable, error::AnvilResult};
use super::{Loss, Reduction, apply_reduction};

/// Mean Squared Error Loss
pub struct MSELoss {
    reduction: Reduction,
}

impl MSELoss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Loss for MSELoss {
    fn forward(&self, predictions: &Variable<f32, 2>, targets: &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>> {
        let diff = predictions.sub(targets)?;
        let squared = diff.mul(&diff)?;
        apply_reduction(squared, self.reduction)
    }
    
    fn name(&self) -> &'static str { "MSELoss" }
    fn reduction(&self) -> Reduction { self.reduction }
}

/// Mean Absolute Error Loss
pub struct MAELoss {
    reduction: Reduction,
}

impl MAELoss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }
}

impl Loss for MAELoss {
    fn forward(&self, predictions: &Variable<f32, 2>, targets: &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>> {
        let diff = predictions.sub(targets)?;
        // In practice, this would compute abs(diff)
        apply_reduction(diff, self.reduction)
    }
    
    fn name(&self) -> &'static str { "MAELoss" }
    fn reduction(&self) -> Reduction { self.reduction }
}

/// Huber Loss (Smooth L1 Loss)
pub struct HuberLoss {
    delta: f32,
    reduction: Reduction,
}

impl HuberLoss {
    pub fn new(delta: f32, reduction: Reduction) -> Self {
        Self { delta, reduction }
    }
}

impl Loss for HuberLoss {
    fn forward(&self, predictions: &Variable<f32, 2>, targets: &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>> {
        let diff = predictions.sub(targets)?;
        // Huber loss implementation would go here
        apply_reduction(diff, self.reduction)
    }
    
    fn name(&self) -> &'static str { "HuberLoss" }
    fn reduction(&self) -> Reduction { self.reduction }
}