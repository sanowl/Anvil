use std::fmt;
use thiserror::Error;
use serde_json;

/// Main error type for the Anvil framework
#[derive(Error, Debug, Clone)]
pub enum AnvilError {
    /// Shape-related errors with detailed suggestions
    #[error("Shape error: {0}")]
    ShapeError(String),

    /// Device-related errors
    #[error("Device error: {0}")]
    DeviceError(String),

    /// Memory-related errors
    #[error("Memory error: {0}")]
    MemoryError(String),

    /// Operation errors with context
    #[error("Operation error: {operation} - {message}")]
    OperationError {
        operation: String,
        message: String,
    },

    /// Compilation errors for compile-time verification
    #[error("Compilation error: {0}")]
    CompilationError(String),

    /// GPU-related errors
    #[error("GPU error: {0}")]
    GpuError(String),

    /// Serialization errors
    #[error("Serialization error: {0}")]
    SerializationError(String),

    /// Configuration errors
    #[error("Configuration error: {0}")]
    ConfigurationError(String),

    /// Distributed training errors
    #[error("Distributed error: {0}")]
    DistributedError(String),

    /// Quantization errors
    #[error("Quantization error: {0}")]
    QuantizationError(String),

    /// Verification errors for formal verification
    #[error("Verification error: {0}")]
    VerificationError(String),

    /// Internal errors that shouldn't happen
    #[error("Internal error: {0}")]
    InternalError(String),

    /// Data-related errors
    #[error("Invalid data: {0}")]
    InvalidData(String),

    /// Unsupported operation errors
    #[error("Unsupported operation: {0}")]
    UnsupportedOperation(String),

    /// Invalid input errors
    #[error("Invalid input: {0}")]
    InvalidInput(String),

    /// Invalid state errors
    #[error("Invalid state: {0}")]
    InvalidState(String),

    /// IO errors
    #[error("IO error: {0}")]
    IoError(String),

    /// Network errors
    #[error("Network error: {0}")]
    NetworkError(String),

    /// Timeout errors
    #[error("Timeout error: {0}")]
    TimeoutError(String),

    /// Resource errors
    #[error("Resource error: {0}")]
    ResourceError(String),
}

impl AnvilError {
    /// Create a shape error with helpful suggestions
    pub fn shape_error(expected: &str, got: &str, suggestion: Option<&str>) -> Self {
        let message = if let Some(sugg) = suggestion {
            format!("Expected {}, got {}. Suggestion: {}", expected, got, sugg)
        } else {
            format!("Expected {}, got {}", expected, got)
        };
        AnvilError::ShapeError(message)
    }

    /// Create an operation error with context
    pub fn operation_error(operation: &str, message: &str) -> Self {
        AnvilError::OperationError {
            operation: operation.to_string(),
            message: message.to_string(),
        }
    }

    /// Create a device error with helpful context
    pub fn device_error(device: &str, message: &str) -> Self {
        AnvilError::DeviceError(format!("Device '{}': {}", device, message))
    }

    /// Create a memory error with usage information
    pub fn memory_error(required: usize, available: usize, context: &str) -> Self {
        AnvilError::MemoryError(format!(
            "Insufficient memory: required {} bytes, available {} bytes. Context: {}",
            required, available, context
        ))
    }

    /// Create a compilation error with shape mismatch details
    pub fn compilation_error(expected_shape: &str, actual_shape: &str, operation: &str) -> Self {
        AnvilError::CompilationError(format!(
            "Shape mismatch in {}: expected {}, got {}. This error was caught at compile time!",
            operation, expected_shape, actual_shape
        ))
    }
}

impl From<std::io::Error> for AnvilError {
    fn from(err: std::io::Error) -> Self {
        AnvilError::IoError(err.to_string())
    }
}

impl From<serde_json::Error> for AnvilError {
    fn from(err: serde_json::Error) -> Self {
        AnvilError::SerializationError(err.to_string())
    }
}

/// Result type for Anvil operations
pub type AnvilResult<T> = Result<T, AnvilError>;

/// Error context for providing additional debugging information
#[derive(Debug, Clone)]
pub struct ErrorContext {
    pub operation: String,
    pub tensor_shapes: Vec<String>,
    pub device_info: String,
    pub memory_usage: Option<String>,
    pub suggestions: Vec<String>,
}

impl ErrorContext {
    pub fn new(operation: &str) -> Self {
        Self {
            operation: operation.to_string(),
            tensor_shapes: Vec::new(),
            device_info: String::new(),
            memory_usage: None,
            suggestions: Vec::new(),
        }
    }

    pub fn with_shape(mut self, shape: &str) -> Self {
        self.tensor_shapes.push(shape.to_string());
        self
    }

    pub fn with_device(mut self, device: &str) -> Self {
        self.device_info = device.to_string();
        self
    }

    pub fn with_memory(mut self, memory: &str) -> Self {
        self.memory_usage = Some(memory.to_string());
        self
    }

    pub fn with_suggestion(mut self, suggestion: &str) -> Self {
        self.suggestions.push(suggestion.to_string());
        self
    }

    pub fn to_error_message(&self) -> String {
        let mut message = format!("Operation: {}", self.operation);
        
        if !self.tensor_shapes.is_empty() {
            message.push_str(&format!("\nTensor shapes: {}", self.tensor_shapes.join(", ")));
        }
        
        if !self.device_info.is_empty() {
            message.push_str(&format!("\nDevice: {}", self.device_info));
        }
        
        if let Some(ref memory) = self.memory_usage {
            message.push_str(&format!("\nMemory: {}", memory));
        }
        
        if !self.suggestions.is_empty() {
            message.push_str("\nSuggestions:");
            for suggestion in &self.suggestions {
                message.push_str(&format!("\n  - {}", suggestion));
            }
        }
        
        message
    }
}

/// Helper trait for adding context to errors
pub trait WithContext<T> {
    fn with_context<F>(self, f: F) -> AnvilResult<T>
    where
        F: FnOnce() -> ErrorContext;
}

impl<T> WithContext<T> for AnvilResult<T> {
    fn with_context<F>(self, f: F) -> AnvilResult<T>
    where
        F: FnOnce() -> ErrorContext,
    {
        self.map_err(|e| {
            let context = f();
            match e {
                AnvilError::ShapeError(msg) => {
                    AnvilError::ShapeError(format!("{}\nContext: {}", msg, context.to_error_message()))
                }
                AnvilError::OperationError { operation, message } => {
                    AnvilError::OperationError {
                        operation,
                        message: format!("{}\nContext: {}", message, context.to_error_message()),
                    }
                }
                AnvilError::InvalidData(msg) => {
                    AnvilError::InvalidData(format!("{}\nContext: {}", msg, context.to_error_message()))
                }
                AnvilError::UnsupportedOperation(msg) => {
                    AnvilError::UnsupportedOperation(format!("{}\nContext: {}", msg, context.to_error_message()))
                }
                _ => e,
            }
        })
    }
}

/// Error codes for programmatic error handling
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorCode {
    ShapeMismatch,
    DeviceNotFound,
    OutOfMemory,
    InvalidOperation,
    CompilationFailed,
    GpuError,
    SerializationFailed,
    ConfigInvalid,
    DistributedError,
    QuantizationError,
    VerificationFailed,
    InternalError,
}

impl AnvilError {
    /// Get the error code for this error
    pub fn code(&self) -> ErrorCode {
        match self {
            AnvilError::ShapeError(_) => ErrorCode::ShapeMismatch,
            AnvilError::DeviceError(_) => ErrorCode::DeviceNotFound,
            AnvilError::MemoryError(_) => ErrorCode::OutOfMemory,
            AnvilError::OperationError { .. } => ErrorCode::InvalidOperation,
            AnvilError::CompilationError(_) => ErrorCode::CompilationFailed,
            AnvilError::GpuError(_) => ErrorCode::GpuError,
            AnvilError::SerializationError(_) => ErrorCode::SerializationFailed,
            AnvilError::ConfigurationError(_) => ErrorCode::ConfigInvalid,
            AnvilError::DistributedError(_) => ErrorCode::DistributedError,
            AnvilError::QuantizationError(_) => ErrorCode::QuantizationError,
            AnvilError::VerificationError(_) => ErrorCode::VerificationFailed,
            AnvilError::InternalError(_) => ErrorCode::InternalError,
            AnvilError::InvalidData(_) => ErrorCode::InvalidOperation,
            AnvilError::UnsupportedOperation(_) => ErrorCode::InvalidOperation,
            AnvilError::InvalidInput(_) => ErrorCode::InvalidOperation,
            AnvilError::InvalidState(_) => ErrorCode::InvalidOperation,
            AnvilError::IoError(_) => ErrorCode::InvalidOperation,
            AnvilError::NetworkError(_) => ErrorCode::InvalidOperation,
            AnvilError::TimeoutError(_) => ErrorCode::InvalidOperation,
            AnvilError::ResourceError(_) => ErrorCode::InvalidOperation,
        }
    }

    /// Check if this is a recoverable error
    pub fn is_recoverable(&self) -> bool {
        matches!(
            self.code(),
            ErrorCode::OutOfMemory | ErrorCode::DeviceNotFound | ErrorCode::ConfigInvalid
        )
    }

    /// Get a user-friendly error message
    pub fn user_message(&self) -> String {
        match self {
            AnvilError::ShapeError(msg) => {
                format!("Shape Error: {}\n\nThis usually means the dimensions of your tensors don't match for the operation you're trying to perform. Check the shapes of your input tensors and make sure they're compatible.", msg)
            }
            AnvilError::DeviceError(msg) => {
                format!("Device Error: {}\n\nThis could mean the GPU is not available, or there's an issue with the device configuration. Try using CPU mode or check your GPU drivers.", msg)
            }
            AnvilError::MemoryError(msg) => {
                format!("Memory Error: {}\n\nYour model or data is too large for the available memory. Try reducing batch size, using gradient checkpointing, or using a smaller model.", msg)
            }
            AnvilError::OperationError { operation, message } => {
                format!("Operation Error in '{}': {}\n\nThis operation failed. Check your input data and operation parameters.", operation, message)
            }
            AnvilError::CompilationError(msg) => {
                format!("Compilation Error: {}\n\nThis error was caught at compile time! The framework detected a problem before running your code. This is a feature of Anvil's compile-time safety.", msg)
            }
            AnvilError::InvalidInput(msg) => {
                format!("Invalid Input: {}\n\nThe input provided is not valid for this operation. Please check your arguments and data.", msg)
            }
            AnvilError::InvalidState(msg) => {
                format!("Invalid State: {}\n\nThe operation could not be performed because the system is in an invalid state. Please check your workflow.", msg)
            }
            _ => self.to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_error() {
        let error = AnvilError::shape_error("[32, 256]", "[32, 128]", Some("Did you mean to use a different linear layer?"));
        assert!(error.to_string().contains("Expected [32, 256], got [32, 128]"));
        assert!(error.to_string().contains("Did you mean to use a different linear layer?"));
    }

    #[test]
    fn test_error_context() {
        let context = ErrorContext::new("matmul")
            .with_shape("[32, 256]")
            .with_shape("[128, 512]")
            .with_device("CUDA:0")
            .with_suggestion("Check if your linear layers have matching dimensions")
            .with_suggestion("Consider using tensor.view() to reshape");

        let message = context.to_error_message();
        assert!(message.contains("Operation: matmul"));
        assert!(message.contains("Tensor shapes: [32, 256], [128, 512]"));
        assert!(message.contains("Device: CUDA:0"));
        assert!(message.contains("Check if your linear layers have matching dimensions"));
    }

    #[test]
    fn test_error_codes() {
        let shape_error = AnvilError::ShapeError("test".to_string());
        assert_eq!(shape_error.code(), ErrorCode::ShapeMismatch);
        assert!(!shape_error.is_recoverable());

        let memory_error = AnvilError::MemoryError("test".to_string());
        assert_eq!(memory_error.code(), ErrorCode::OutOfMemory);
        assert!(memory_error.is_recoverable());
    }

    #[test]
    fn test_user_message() {
        let error = AnvilError::CompilationError("Shape mismatch in matmul".to_string());
        let message = error.user_message();
        assert!(message.contains("Compilation Error"));
        assert!(message.contains("compile time"));
        assert!(message.contains("feature of Anvil's compile-time safety"));
    }
} 