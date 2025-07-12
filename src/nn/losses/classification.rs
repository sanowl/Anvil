//! Classification loss functions with automatic differentiation

use crate::{
    autograd::Variable,
    error::{AnvilError, AnvilResult},
};
use super::{Loss, Reduction, apply_reduction, utils::*};

/// Cross-entropy loss for multi-class classification
pub struct CrossEntropyLoss {
    reduction: Reduction,
    label_smoothing: Option<f32>,
    ignore_index: Option<i64>,
    weight: Option<Variable<f32, 1>>,
}

impl CrossEntropyLoss {
    pub fn new(reduction: Reduction, label_smoothing: Option<f32>) -> Self {
        Self {
            reduction,
            label_smoothing,
            ignore_index: None,
            weight: None,
        }
    }
    
    pub fn with_ignore_index(mut self, ignore_index: i64) -> Self {
        self.ignore_index = Some(ignore_index);
        self
    }
    
    pub fn with_class_weights(mut self, weights: Variable<f32, 1>) -> Self {
        self.weight = Some(weights);
        self
    }
    
    /// Compute cross-entropy loss with numerical stability
    fn compute_cross_entropy(
        &self,
        logits: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 2>> {
        // Apply softmax to get probabilities
        let log_probs = logits.log_softmax(1)?;
        
        // Apply label smoothing if specified
        let smoothed_targets = if let Some(smoothing) = self.label_smoothing {
            self.apply_label_smoothing(targets, smoothing)?
        } else {
            targets.clone()
        };
        
        // Compute negative log-likelihood
        let nll = self.negative_log_likelihood(&log_probs, &smoothed_targets)?;
        
        // Apply class weights if specified
        if let Some(weights) = &self.weight {
            self.apply_class_weights(&nll, &smoothed_targets, weights)
        } else {
            Ok(nll)
        }
    }
    
    fn apply_label_smoothing(
        &self,
        targets: &Variable<f32, 2>,
        smoothing: f32,
    ) -> AnvilResult<Variable<f32, 2>> {
        let num_classes = targets.shape().dims[1];
        let smooth_value = smoothing / num_classes as f32;
        let target_value = 1.0 - smoothing + smooth_value;
        
        // Create smoothed targets: (1 - ε) * y + ε / K
        // where ε is smoothing factor, K is number of classes
        
        // This is a placeholder implementation
        // In practice, would modify the one-hot targets
        Ok(targets.clone())
    }
    
    fn negative_log_likelihood(
        &self,
        log_probs: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 2>> {
        // NLL = -sum(targets * log_probs)
        let product = log_probs.mul(targets)?;
        
        // Sum over class dimension and negate
        // This is simplified - would need proper reduction over class dimension
        Ok(product)
    }
    
    fn apply_class_weights(
        &self,
        nll: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
        weights: &Variable<f32, 1>,
    ) -> AnvilResult<Variable<f32, 2>> {
        // Apply per-class weights based on target class
        // This is a placeholder implementation
        Ok(nll.clone())
    }
}

impl Loss for CrossEntropyLoss {
    fn forward(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 0>> {
        let loss_per_sample = self.compute_cross_entropy(predictions, targets)?;
        apply_reduction(loss_per_sample, self.reduction)
    }
    
    fn name(&self) -> &'static str {
        "CrossEntropyLoss"
    }
    
    fn reduction(&self) -> Reduction {
        self.reduction
    }
    
    fn requires_one_hot(&self) -> bool {
        true
    }
}

/// Binary cross-entropy loss
pub struct BCELoss {
    reduction: Reduction,
    pos_weight: Option<Variable<f32, 1>>,
    eps: f32,
}

impl BCELoss {
    pub fn new(reduction: Reduction, pos_weight: Option<Variable<f32, 1>>) -> Self {
        Self {
            reduction,
            pos_weight,
            eps: 1e-7, // For numerical stability
        }
    }
    
    fn compute_bce(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 2>> {
        // Clamp predictions for numerical stability
        let clamped_preds = clamp_for_log(predictions, self.eps, 1.0 - self.eps)?;
        
        // BCE = -[y * log(p) + (1-y) * log(1-p)]
        let log_p = safe_log(&clamped_preds, self.eps)?;
        let log_1_minus_p = safe_log(&clamped_preds, self.eps)?; // Placeholder - should be log(1-p)
        
        let pos_term = targets.mul(&log_p)?;
        // let neg_term = (1 - targets) * log(1 - predictions)
        let neg_term = pos_term.clone(); // Placeholder
        
        let bce = pos_term.add(&neg_term)?;
        
        // Apply positive weight if specified
        if let Some(pos_weight) = &self.pos_weight {
            self.apply_pos_weight(&bce, targets, pos_weight)
        } else {
            Ok(bce)
        }
    }
    
    fn apply_pos_weight(
        &self,
        bce: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
        pos_weight: &Variable<f32, 1>,
    ) -> AnvilResult<Variable<f32, 2>> {
        // Apply positive class weighting
        // weighted_bce = pos_weight * y * bce + (1 - y) * bce
        Ok(bce.clone()) // Placeholder
    }
}

impl Loss for BCELoss {
    fn forward(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 0>> {
        let loss_per_sample = self.compute_bce(predictions, targets)?;
        apply_reduction(loss_per_sample, self.reduction)
    }
    
    fn name(&self) -> &'static str {
        "BCELoss"
    }
    
    fn reduction(&self) -> Reduction {
        self.reduction
    }
    
    fn requires_probabilities(&self) -> bool {
        true
    }
}

/// Focal loss for addressing class imbalance
pub struct FocalLoss {
    alpha: f32,
    gamma: f32,
    reduction: Reduction,
}

impl FocalLoss {
    pub fn new(alpha: f32, gamma: f32, reduction: Reduction) -> Self {
        Self {
            alpha,
            gamma,
            reduction,
        }
    }
    
    fn compute_focal_loss(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 2>> {
        // Get probabilities
        let probs = safe_softmax(predictions, 1)?;
        
        // Compute cross-entropy
        let log_probs = safe_log(&probs, 1e-8)?;
        let ce = targets.mul(&log_probs)?;
        
        // Compute focal weight: (1 - p)^γ
        // where p is the probability of the true class
        let pt = targets.mul(&probs)?; // p for true class
        
        // focal_weight = (1 - pt)^gamma
        // This is simplified - would need proper power operation
        let focal_weight = pt.clone(); // Placeholder
        
        // Apply alpha weighting
        let alpha_weight = self.alpha;
        
        // Focal loss = -α * (1-pt)^γ * CE
        let focal_loss = ce.mul(&focal_weight)?;
        
        Ok(focal_loss)
    }
}

impl Loss for FocalLoss {
    fn forward(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 0>> {
        let loss_per_sample = self.compute_focal_loss(predictions, targets)?;
        apply_reduction(loss_per_sample, self.reduction)
    }
    
    fn name(&self) -> &'static str {
        "FocalLoss"
    }
    
    fn reduction(&self) -> Reduction {
        self.reduction
    }
    
    fn requires_one_hot(&self) -> bool {
        true
    }
}

/// Dice loss for segmentation tasks
pub struct DiceLoss {
    smooth: f32,
    reduction: Reduction,
}

impl DiceLoss {
    pub fn new(smooth: f32, reduction: Reduction) -> Self {
        Self {
            smooth,
            reduction,
        }
    }
    
    fn compute_dice_loss(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 2>> {
        // Apply softmax/sigmoid to get probabilities
        let probs = safe_softmax(predictions, 1)?;
        
        // Compute Dice coefficient
        let intersection = probs.mul(targets)?;
        let union = probs.add(targets)?;
        
        // Dice = 2 * |intersection| / (|A| + |B|)
        // Dice Loss = 1 - Dice
        
        // This is simplified - would need proper sum operations
        let dice_coeff = intersection.clone(); // Placeholder
        let dice_loss = dice_coeff; // Placeholder: 1 - dice_coeff
        
        Ok(dice_loss)
    }
}

impl Loss for DiceLoss {
    fn forward(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 0>> {
        let loss_per_sample = self.compute_dice_loss(predictions, targets)?;
        apply_reduction(loss_per_sample, self.reduction)
    }
    
    fn name(&self) -> &'static str {
        "DiceLoss"
    }
    
    fn reduction(&self) -> Reduction {
        self.reduction
    }
    
    fn requires_probabilities(&self) -> bool {
        true
    }
}

/// Label smoothing cross-entropy loss
pub struct LabelSmoothingLoss {
    smoothing: f32,
    reduction: Reduction,
}

impl LabelSmoothingLoss {
    pub fn new(smoothing: f32, reduction: Reduction) -> Self {
        Self {
            smoothing,
            reduction,
        }
    }
}

impl Loss for LabelSmoothingLoss {
    fn forward(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 0>> {
        let ce_loss = CrossEntropyLoss::new(self.reduction, Some(self.smoothing));
        ce_loss.forward(predictions, targets)
    }
    
    fn name(&self) -> &'static str {
        "LabelSmoothingLoss"
    }
    
    fn reduction(&self) -> Reduction {
        self.reduction
    }
    
    fn requires_one_hot(&self) -> bool {
        true
    }
}

/// Categorical hinge loss (multiclass SVM loss)
pub struct CategoricalHingeLoss {
    margin: f32,
    reduction: Reduction,
}

impl CategoricalHingeLoss {
    pub fn new(margin: f32, reduction: Reduction) -> Self {
        Self {
            margin,
            reduction,
        }
    }
    
    fn compute_hinge_loss(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 2>> {
        // Categorical hinge loss: max(0, margin - (true_class_score - max(other_class_scores)))
        
        // Get score for true class
        let true_class_scores = predictions.mul(targets)?;
        
        // Get max score for other classes
        // This is simplified - would need proper masking and max operations
        let other_class_scores = predictions.clone(); // Placeholder
        
        // Compute hinge loss
        let hinge = true_class_scores.sub(&other_class_scores)?;
        
        // Apply margin and ReLU
        // loss = max(0, margin - hinge)
        let loss = hinge.relu()?; // Placeholder - should be max(0, margin - hinge)
        
        Ok(loss)
    }
}

impl Loss for CategoricalHingeLoss {
    fn forward(
        &self,
        predictions: &Variable<f32, 2>,
        targets: &Variable<f32, 2>,
    ) -> AnvilResult<Variable<f32, 0>> {
        let loss_per_sample = self.compute_hinge_loss(predictions, targets)?;
        apply_reduction(loss_per_sample, self.reduction)
    }
    
    fn name(&self) -> &'static str {
        "CategoricalHingeLoss"
    }
    
    fn reduction(&self) -> Reduction {
        self.reduction
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};
    
    #[test]
    fn test_cross_entropy_loss() {
        let loss = CrossEntropyLoss::new(Reduction::Mean, None);
        assert_eq!(loss.name(), "CrossEntropyLoss");
        assert_eq!(loss.reduction(), Reduction::Mean);
        assert!(loss.requires_one_hot());
    }
    
    #[test]
    fn test_bce_loss() {
        let loss = BCELoss::new(Reduction::Mean, None);
        assert_eq!(loss.name(), "BCELoss");
        assert!(loss.requires_probabilities());
    }
    
    #[test]
    fn test_focal_loss() {
        let loss = FocalLoss::new(1.0, 2.0, Reduction::Mean);
        assert_eq!(loss.name(), "FocalLoss");
        assert_eq!(loss.reduction(), Reduction::Mean);
    }
    
    #[test]
    fn test_dice_loss() {
        let loss = DiceLoss::new(1e-8, Reduction::Mean);
        assert_eq!(loss.name(), "DiceLoss");
        assert!(loss.requires_probabilities());
    }
    
    #[test]
    fn test_label_smoothing() {
        let loss = LabelSmoothingLoss::new(0.1, Reduction::Mean);
        assert_eq!(loss.name(), "LabelSmoothingLoss");
        assert!(loss.requires_one_hot());
    }
    
    #[test]
    fn test_hinge_loss() {
        let loss = CategoricalHingeLoss::new(1.0, Reduction::Mean);
        assert_eq!(loss.name(), "CategoricalHingeLoss");
    }
}