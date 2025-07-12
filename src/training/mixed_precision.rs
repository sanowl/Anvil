//! Mixed precision training support

use crate::{
    autograd::Variable,
    error::{AnvilError, AnvilResult},
};

/// Mixed precision training configuration
#[derive(Debug, Clone)]
pub struct MixedPrecisionConfig {
    pub enabled: bool,
    pub loss_scale: f32,
    pub growth_factor: f32,
    pub backoff_factor: f32,
    pub growth_interval: usize,
    pub dynamic_scaling: bool,
}

impl Default for MixedPrecisionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            loss_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            dynamic_scaling: true,
        }
    }
}

/// Gradient scaler for mixed precision training
pub struct GradientScaler {
    config: MixedPrecisionConfig,
    current_scale: f32,
    growth_tracker: usize,
    has_inf_or_nan: bool,
}

impl GradientScaler {
    pub fn new(config: MixedPrecisionConfig) -> Self {
        Self {
            current_scale: config.loss_scale,
            config,
            growth_tracker: 0,
            has_inf_or_nan: false,
        }
    }
    
    /// Scale the loss for backward pass
    pub fn scale_loss(&self, loss: Variable<f32, 0>) -> AnvilResult<Variable<f32, 0>> {
        if !self.config.enabled {
            return Ok(loss);
        }
        
        // Scale the loss by current scale factor
        // In practice, this would multiply the loss tensor by the scale
        Ok(loss)
    }
    
    /// Unscale gradients after backward pass
    pub fn unscale_gradients(&mut self, gradients: Vec<Variable<f32, 2>>) -> AnvilResult<Vec<Variable<f32, 2>>> {
        if !self.config.enabled {
            return Ok(gradients);
        }
        
        // Check for inf/nan in gradients
        self.has_inf_or_nan = self.check_inf_nan(&gradients);
        
        if self.has_inf_or_nan {
            // Skip this step - gradients are invalid
            return Ok(gradients);
        }
        
        // Unscale gradients
        let unscaled_gradients = gradients.into_iter().map(|grad| {
            // In practice, this would divide the gradient by the current scale
            grad
        }).collect();
        
        Ok(unscaled_gradients)
    }
    
    /// Update the loss scale based on gradient validity
    pub fn update_scale(&mut self) {
        if !self.config.enabled || !self.config.dynamic_scaling {
            return;
        }
        
        if self.has_inf_or_nan {
            // Reduce scale due to overflow
            self.current_scale *= self.config.backoff_factor;
            self.current_scale = self.current_scale.max(1.0);
            self.growth_tracker = 0;
            println!("Reduced loss scale to {} due to gradient overflow", self.current_scale);
        } else {
            // Increment growth tracker
            self.growth_tracker += 1;
            
            if self.growth_tracker >= self.config.growth_interval {
                // Increase scale
                self.current_scale *= self.config.growth_factor;
                self.current_scale = self.current_scale.min(65536.0); // Cap at reasonable value
                self.growth_tracker = 0;
                println!("Increased loss scale to {}", self.current_scale);
            }
        }
        
        // Reset the inf/nan flag
        self.has_inf_or_nan = false;
    }
    
    /// Check if gradients contain inf or nan values
    fn check_inf_nan(&self, gradients: &[Variable<f32, 2>]) -> bool {
        for grad in gradients {
            let grad_data = grad.as_slice::<f32>();
            for &val in grad_data {
                if val.is_infinite() || val.is_nan() {
                    return true;
                }
            }
        }
        false
    }
    
    /// Get current loss scale
    pub fn get_scale(&self) -> f32 {
        if self.config.enabled {
            self.current_scale
        } else {
            1.0
        }
    }
    
    /// Set loss scale manually
    pub fn set_scale(&mut self, scale: f32) {
        self.current_scale = scale;
        self.growth_tracker = 0;
    }
    
    /// Check if last step had gradient overflow
    pub fn had_overflow(&self) -> bool {
        self.has_inf_or_nan
    }
    
    /// Enable or disable mixed precision
    pub fn set_enabled(&mut self, enabled: bool) {
        self.config.enabled = enabled;
        if !enabled {
            self.current_scale = 1.0;
            self.growth_tracker = 0;
        }
    }
}

/// Automatic mixed precision context manager
pub struct AutocastContext {
    enabled: bool,
    dtype: DataType,
}

#[derive(Debug, Clone, Copy)]
pub enum DataType {
    Float32,
    Float16,
    BFloat16,
}

impl AutocastContext {
    pub fn new(enabled: bool, dtype: DataType) -> Self {
        Self { enabled, dtype }
    }
    
    /// Execute a closure with automatic casting
    pub fn with_autocast<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        if self.enabled {
            // In practice, this would set thread-local state for automatic casting
            // and restore it after the closure execution
            println!("Running with autocast enabled ({:?})", self.dtype);
        }
        
        f()
    }
    
    /// Check if autocast is enabled
    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
    
    /// Get the autocast dtype
    pub fn dtype(&self) -> DataType {
        self.dtype
    }
}

/// Mixed precision utilities
pub mod utils {
    use super::*;
    
    /// Convert tensor to half precision
    pub fn to_half_precision(tensor: &Variable<f32, 2>) -> AnvilResult<Variable<f32, 2>> {
        // In practice, this would convert f32 to f16
        // For now, return a copy
        Ok(tensor.clone())
    }
    
    /// Convert tensor to full precision
    pub fn to_full_precision(tensor: &Variable<f32, 2>) -> AnvilResult<Variable<f32, 2>> {
        // In practice, this would convert f16 to f32
        // For now, return a copy
        Ok(tensor.clone())
    }
    
    /// Check if a value can be represented in half precision
    pub fn is_half_precision_safe(value: f32) -> bool {
        // Check if value is within f16 range
        let abs_val = value.abs();
        abs_val <= 65504.0 && (abs_val == 0.0 || abs_val >= 6.103515625e-5)
    }
    
    /// Clamp value to half precision range
    pub fn clamp_to_half_precision(value: f32) -> f32 {
        if value.is_nan() {
            return value;
        }
        
        let sign = if value < 0.0 { -1.0 } else { 1.0 };
        let abs_val = value.abs();
        
        if abs_val > 65504.0 {
            sign * 65504.0
        } else if abs_val < 6.103515625e-5 && abs_val != 0.0 {
            sign * 6.103515625e-5
        } else {
            value
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};
    
    #[test]
    fn test_gradient_scaler() {
        let config = MixedPrecisionConfig::default();
        let mut scaler = GradientScaler::new(config);
        
        assert_eq!(scaler.get_scale(), 65536.0);
        
        // Simulate a few successful steps
        for _ in 0..5 {
            scaler.has_inf_or_nan = false;
            scaler.update_scale();
        }
        
        // Scale should remain the same (not enough steps for growth)
        assert_eq!(scaler.get_scale(), 65536.0);
        
        // Simulate gradient overflow
        scaler.has_inf_or_nan = true;
        scaler.update_scale();
        
        // Scale should be reduced
        assert_eq!(scaler.get_scale(), 32768.0);
    }
    
    #[test]
    fn test_autocast_context() {
        let autocast = AutocastContext::new(true, DataType::Float16);
        assert!(autocast.is_enabled());
        
        let result = autocast.with_autocast(|| {
            42
        });
        
        assert_eq!(result, 42);
    }
    
    #[test]
    fn test_half_precision_utils() {
        use utils::*;
        
        assert!(is_half_precision_safe(1.0));
        assert!(is_half_precision_safe(-1.0));
        assert!(!is_half_precision_safe(100000.0));
        assert!(!is_half_precision_safe(1e-10));
        
        assert_eq!(clamp_to_half_precision(100000.0), 65504.0);
        assert_eq!(clamp_to_half_precision(-100000.0), -65504.0);
    }
}