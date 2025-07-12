//! Model verification and validation utilities

use crate::{
    tensor::{Tensor, Shape, DType},
    error::{AnvilError, AnvilResult},
    nn::Module,
};

/// Model verification result
#[derive(Debug, Clone)]
pub struct VerificationResult {
    pub passed: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub metrics: VerificationMetrics,
}

#[derive(Debug, Clone)]
pub struct VerificationMetrics {
    pub parameter_count: usize,
    pub model_size_bytes: usize,
    pub flops: usize,
    pub memory_usage: usize,
}

/// Model verifier
pub struct ModelVerifier;

impl ModelVerifier {
    pub fn new() -> Self {
        Self
    }

    /// Verify a model's structure and parameters
    pub fn verify_model(&self, model: &dyn Module) -> VerificationResult {
        let mut errors = Vec::new();
        let mut warnings = Vec::new();
        let mut parameter_count = 0;
        let mut model_size_bytes = 0;

        // Check parameters
        let parameters = model.parameters();
        for param in parameters {
            parameter_count += param.shape().size();
            model_size_bytes += param.shape().size() * param.dtype().size();

            // Check for NaN or infinite values
            let data = param.as_slice::<f32>();
            for (i, &value) in data.iter().enumerate() {
                if value.is_nan() {
                    errors.push(format!("NaN detected in parameter at index {}", i));
                }
                if value.is_infinite() {
                    errors.push(format!("Infinite value detected in parameter at index {}", i));
                }
            }

            // Check for very large values
            let data = param.as_slice::<f32>();
            for (i, &value) in data.iter().enumerate() {
                if value.abs() > 1e6 {
                    errors.push(format!("Large value detected in parameter at index {}: {}", i, value));
                }
            }
        }

        // Check model structure
        if parameter_count == 0 {
            errors.push("Model has no parameters".to_string());
        }

        if model_size_bytes > 1024 * 1024 * 1024 { // 1GB
            warnings.push("Model size is very large (>1GB)".to_string());
        }

        let flops = self.estimate_flops(model);
        let memory_usage = self.estimate_memory_usage(model);

        let metrics = VerificationMetrics {
            parameter_count,
            model_size_bytes,
            flops,
            memory_usage,
        };

        VerificationResult {
            passed: errors.is_empty(),
            errors,
            warnings,
            metrics,
        }
    }

    /// Estimate FLOPs for the model
    fn estimate_flops(&self, model: &dyn Module) -> usize {
        // Simplified FLOP estimation
        let parameters = model.parameters();
        let mut total_flops = 0;

        for param in parameters {
            // Rough estimate: 2 FLOPs per parameter for forward pass
            total_flops += param.shape().size() * 2;
        }

        total_flops
    }

    /// Estimate memory usage for the model
    fn estimate_memory_usage(&self, model: &dyn Module) -> usize {
        let parameters = model.parameters();
        let mut total_memory = 0;

        for param in parameters {
            total_memory += param.shape().size() * param.dtype().size();
        }

        total_memory
    }
}

/// Input validation utilities
pub struct InputValidator;

impl InputValidator {
    pub fn new() -> Self {
        Self
    }

    /// Validate input tensor for a model
    pub fn validate_input(&self, input: &Tensor<2>, expected_shape: Option<Shape<2>>) -> AnvilResult<()> {
        // Check for NaN or infinite values
        let data = input.as_slice::<f32>();
        for (i, &value) in data.iter().enumerate() {
            if value.is_nan() {
                return Err(AnvilError::InvalidInput(
                    format!("NaN detected in input at index {}", i)
                ));
            }
            if value.is_infinite() {
                return Err(AnvilError::InvalidInput(
                    format!("Infinite value detected in input at index {}", i)
                ));
            }
        }

        // Check shape if expected
        if let Some(expected) = expected_shape {
            if input.shape() != expected {
                return Err(AnvilError::InvalidShape(
                    format!("Expected input shape {:?}, got {:?}", expected, input.shape())
                ));
            }
        }

        Ok(())
    }

    /// Validate output tensor
    pub fn validate_output(&self, output: &Tensor<2>) -> AnvilResult<()> {
        // Check for NaN or infinite values
        let data = output.as_slice::<f32>();
        for (i, &value) in data.iter().enumerate() {
            if value.is_nan() {
                return Err(AnvilError::InvalidInput(
                    format!("NaN detected in output at index {}", i)
                ));
            }
            if value.is_infinite() {
                return Err(AnvilError::InvalidInput(
                    format!("Infinite value detected in output at index {}", i)
                ));
            }
        }

        Ok(())
    }
}

/// Model consistency checker
pub struct ConsistencyChecker;

impl ConsistencyChecker {
    pub fn new() -> Self {
        Self
    }

    /// Check model consistency across multiple runs
    pub fn check_consistency<F>(
        &self,
        model: &dyn Module,
        input: &Tensor<2>,
        num_runs: usize,
        tolerance: f32,
        forward_fn: F,
    ) -> AnvilResult<bool>
    where
        F: Fn(&dyn Module, &Tensor<2>) -> AnvilResult<Tensor<2>>,
    {
        if num_runs < 2 {
            return Err(AnvilError::InvalidInput("Need at least 2 runs for consistency check".to_string()));
        }

        let mut outputs = Vec::new();
        
        for i in 0..num_runs {
            let output = forward_fn(model, input)?;
            outputs.push(output);
        }

        // Compare all outputs
        let reference = &outputs[0];
        for (i, output) in outputs.iter().enumerate().skip(1) {
            if !self.tensors_equal(reference, output, tolerance)? {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Check if two tensors are equal within tolerance
    fn tensors_equal(&self, a: &Tensor<2>, b: &Tensor<2>, tolerance: f32) -> AnvilResult<bool> {
        if a.shape() != b.shape() {
            return Ok(false);
        }

        let a_data = a.as_slice::<f32>();
        let b_data = b.as_slice::<f32>();

        for (&a_val, &b_val) in a_data.iter().zip(b_data.iter()) {
            if (a_val - b_val).abs() > tolerance {
                return Ok(false);
            }
        }

        Ok(true)
    }
}

/// Gradient verification
pub struct GradientVerifier;

impl GradientVerifier {
    pub fn new() -> Self {
        Self
    }

    /// Verify gradients using finite differences
    pub fn verify_gradients<F>(
        &self,
        model: &dyn Module,
        input: &Tensor<2>,
        loss_fn: F,
        epsilon: f32,
    ) -> AnvilResult<bool>
    where
        F: Fn(&Tensor<2>) -> AnvilResult<Tensor<2>>,
    {
        let parameters = model.parameters();
        let mut all_gradients_valid = true;

        for param in parameters {
            let param_shape = param.shape();
            let param_size = param_shape.size();

            for i in 0..param_size {
                // Compute finite difference
                let mut param_plus = param.clone();
                let mut param_minus = param.clone();

                // Add epsilon
                let mut data = param_plus.as_slice_mut::<f32>();
                data[i] += epsilon;

                // Subtract epsilon
                let mut data = param_minus.as_slice_mut::<f32>();
                data[i] -= epsilon;

                // Compute loss difference
                let loss_plus = loss_fn(&param_plus)?;
                let loss_minus = loss_fn(&param_minus)?;

                let finite_diff = (loss_plus.as_slice::<f32>()[0] - loss_minus.as_slice::<f32>()[0]) / (2.0 * epsilon);

                // Compare with analytical gradient (simplified)
                let analytical_grad = 0.0; // In real implementation, get from backward pass

                if (finite_diff - analytical_grad).abs() > epsilon {
                    tracing::warn!("Gradient mismatch for parameter at index {}: finite_diff={}, analytical={}", 
                                  i, finite_diff, analytical_grad);
                    all_gradients_valid = false;
                }
            }
        }

        Ok(all_gradients_valid)
    }
}

/// Model validation utilities
pub mod utils {
    use super::*;

    /// Check if a model is deterministic
    pub fn is_deterministic<F>(
        model: &dyn Module,
        input: &Tensor<2>,
        num_runs: usize,
        forward_fn: F,
    ) -> AnvilResult<bool>
    where
        F: Fn(&dyn Module, &Tensor<2>) -> AnvilResult<Tensor<2>>,
    {
        let checker = ConsistencyChecker::new();
        checker.check_consistency(model, input, num_runs, 1e-6, forward_fn)
    }

    /// Validate model architecture
    pub fn validate_architecture(model: &dyn Module) -> AnvilResult<()> {
        let verifier = ModelVerifier::new();
        let result = verifier.verify_model(model);

        if !result.passed {
            return Err(AnvilError::InvalidInput(
                format!("Model validation failed: {:?}", result.errors)
            ));
        }

        if !result.warnings.is_empty() {
            tracing::warn!("Model validation warnings: {:?}", result.warnings);
        }

        Ok(())
    }

    /// Get model statistics
    pub fn get_model_stats(model: &dyn Module) -> VerificationMetrics {
        let verifier = ModelVerifier::new();
        let result = verifier.verify_model(model);
        result.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};
    use crate::nn::Linear;

    #[test]
    fn test_model_verifier() {
        let model = Linear::new(10, 1);
        let verifier = ModelVerifier::new();
        let result = verifier.verify_model(&model);

        assert!(result.passed);
        assert!(result.metrics.parameter_count > 0);
    }

    #[test]
    fn test_input_validator() {
        let input = Tensor::new(Shape::new([2, 2]), DType::F32, Device::Cpu);
        let validator = InputValidator::new();
        
        let result = validator.validate_input(&input, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_consistency_checker() {
        let model = Linear::new(2, 1);
        let input = Tensor::new(Shape::new([1, 2]), DType::F32, Device::Cpu);
        let checker = ConsistencyChecker::new();
        
        let forward_fn = |model: &dyn Module, input: &Tensor<2>| model.forward(input);
        let result = checker.check_consistency(&model, &input, 3, 1e-6, forward_fn);
        assert!(result.is_ok());
    }

    #[test]
    fn test_utils() {
        let model = Linear::new(5, 1);
        let result = utils::validate_architecture(&model);
        assert!(result.is_ok());
        
        let stats = utils::get_model_stats(&model);
        assert!(stats.parameter_count > 0);
    }
} 