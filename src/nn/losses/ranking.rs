//! Ranking and metric learning loss functions

use crate::{autograd::Variable, error::AnvilResult};
use super::{Loss, Reduction, apply_reduction};

/// Triplet Loss for metric learning
pub struct TripletLoss {
    margin: f32,
    reduction: Reduction,
}

impl TripletLoss {
    pub fn new(margin: f32, reduction: Reduction) -> Self {
        Self { margin, reduction }
    }
}

impl Loss for TripletLoss {
    fn forward(&self, predictions: &Variable<f32, 2>, targets: &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>> {
        // Triplet loss implementation would compute distance between anchor, positive, negative
        apply_reduction(predictions.clone(), self.reduction)
    }
    
    fn name(&self) -> &'static str { "TripletLoss" }
    fn reduction(&self) -> Reduction { self.reduction }
}