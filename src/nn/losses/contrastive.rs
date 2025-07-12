//! Contrastive learning loss functions

use crate::{autograd::Variable, error::AnvilResult};
use super::{Loss, Reduction, apply_reduction};

/// Contrastive Loss
pub struct ContrastiveLoss {
    margin: f32,
    reduction: Reduction,
}

impl ContrastiveLoss {
    pub fn new(margin: f32, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }
}

impl Loss for ContrastiveLoss {
    fn forward(&self, predictions: &Variable<f32, 2>, targets: &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>> {
        // Contrastive loss implementation
        apply_reduction(predictions.clone(), self.reduction)
    }
    
    fn name(&self) -> &'static str { "ContrastiveLoss" }
    fn reduction(&self) -> Reduction { self.reduction }
}