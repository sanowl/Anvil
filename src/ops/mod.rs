//! Advanced tensor operations with SIMD, GPU, and optimization support

pub mod core;
pub mod fusion;
pub mod linear;
pub mod activation;
pub mod conv;
pub mod attention;
pub mod optimization;

pub use core::*;
pub use fusion::*;
pub use linear::*;
pub use activation::*;
pub use conv::*;
pub use attention::*;
pub use optimization::*;

// Legacy types for backward compatibility
pub struct FusedOperation;
pub trait TensorOperation {}
pub struct LinearOp;
pub struct ReLUOp;
pub struct DropoutOp;
pub struct ConvOp;
pub struct BatchNormOp;
pub struct AttentionOp; 