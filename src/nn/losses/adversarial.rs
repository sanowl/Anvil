//! Adversarial loss functions for GANs

use crate::{autograd::Variable, error::AnvilResult};
use super::{Loss, Reduction, apply_reduction};

/// Adversarial loss modes
#[derive(Debug, Clone, Copy)]
pub enum AdversarialMode {
    Standard,
    Wasserstein,
    LeastSquares,
}

/// Adversarial Loss for GANs
pub struct AdversarialLoss {
    mode: AdversarialMode,
    reduction: Reduction,
}

impl AdversarialLoss {
    pub fn new(mode: AdversarialMode, reduction: Reduction) -> Self {
        Self { mode, reduction }
    }
}

impl Loss for AdversarialLoss {
    fn forward(&self, predictions: &Variable<f32, 2>, targets: &Variable<f32, 2>) -> AnvilResult<Variable<f32, 0>> {
        match self.mode {
            AdversarialMode::Standard => {
                // Standard GAN loss
                apply_reduction(predictions.clone(), self.reduction)
            },
            AdversarialMode::Wasserstein => {
                // Wasserstein GAN loss
                apply_reduction(predictions.clone(), self.reduction)
            },
            AdversarialMode::LeastSquares => {
                // Least squares GAN loss
                apply_reduction(predictions.clone(), self.reduction)
            },
        }
    }
    
    fn name(&self) -> &'static str { "AdversarialLoss" }
    fn reduction(&self) -> Reduction { self.reduction }
}