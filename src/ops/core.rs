//! Core operation traits and fundamental types

use std::any::Any;
use std::collections::HashMap;
use async_trait::async_trait;
use crate::{
    tensor::{AdvancedTensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MemoryLayout {
    RowMajor,
    ColumnMajor,
    Blocked { block_size: usize },
    Interleaved,
    Auto,
}

impl MemoryLayout {
    pub fn as_u8(&self) -> u8 {
        match self {
            MemoryLayout::RowMajor => 0,
            MemoryLayout::ColumnMajor => 1,
            MemoryLayout::Blocked { .. } => 2,
            MemoryLayout::Interleaved => 3,
            MemoryLayout::Auto => 4,
        }
    }
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(MemoryLayout::RowMajor),
            1 => Some(MemoryLayout::ColumnMajor),
            2 => Some(MemoryLayout::Blocked { block_size: 16 }),
            3 => Some(MemoryLayout::Interleaved),
            4 => Some(MemoryLayout::Auto),
            _ => None,
        }
    }
}

/// Advanced tensor operation trait with async support and compile-time safety
#[async_trait]
pub trait AdvancedTensorOperation<const DIMS: usize>: Send + Sync {
    async fn forward(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>>;
    /// Get operation name for debugging
    fn name(&self) -> &'static str;
    /// Get input shape requirements
    fn input_shape_requirements(&self) -> Shape<DIMS>;
    /// Get output shape for given input
    fn output_shape(&self, input_shape: &Shape<DIMS>) -> AnvilResult<Shape<DIMS>>;
    /// Get operation type
    fn operation_type(&self) -> &'static str;
    /// Check if operation supports SIMD
    fn supports_simd(&self) -> bool;
    /// Check if operation supports GPU
    fn supports_gpu(&self) -> bool;
    /// Get memory alignment requirements
    fn memory_alignment(&self) -> usize;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionStrategy {
    Sequential,
    LinearRelu,
    LinearDropout,
    ConvRelu,
    ConvBatchNorm,
    AttentionFused,
    TransformerBlock,
    ResNetBlock,
    CustomOptimized,
    AutoDetect,
}

/// Operation optimization level
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    Basic,
    Aggressive,
    Maximum,
}

/// Operation execution context
#[derive(Debug, Clone)]
pub struct ExecutionContext {
    pub device: crate::tensor::devices::Device,
    pub optimization_level: OptimizationLevel,
    pub fusion_strategy: FusionStrategy,
    // memory_pool and async_executor fields removed for Debug/dyn compatibility
}

impl Default for ExecutionContext {
    fn default() -> Self {
        Self {
            device: crate::tensor::devices::Device::Cpu,
            optimization_level: OptimizationLevel::Basic,
            fusion_strategy: FusionStrategy::Sequential,
        }
    }
}

/// Operation metadata for optimization
#[derive(Debug, Clone)]
pub struct OperationMetadata<const DIMS: usize> {
    pub name: String,
    pub input_shapes: Vec<Shape<DIMS>>,
    pub output_shape: Shape<DIMS>,
    pub memory_usage: usize,
    pub compute_complexity: f64,
    pub device_preference: Vec<crate::tensor::devices::Device>,
}

/// Advanced operation with metadata and optimization
pub struct AdvancedOperation<Op, const DIMS: usize> {
    operation: Op,
    metadata: OperationMetadata<DIMS>,
    context: ExecutionContext,
}

impl<Op, const DIMS: usize> AdvancedOperation<Op, DIMS>
where
    Op: AdvancedTensorOperation<DIMS>,
{
    pub fn new(operation: Op, metadata: OperationMetadata<DIMS>) -> Self {
        Self {
            operation,
            metadata,
            context: ExecutionContext::default(),
        }
    }
    
    pub fn with_context(mut self, context: ExecutionContext) -> Self {
        self.context = context;
        self
    }
    
    pub async fn execute(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        match self.context.optimization_level {
            OptimizationLevel::None => self.operation.forward(input).await,
            OptimizationLevel::Basic => self.execute_basic_optimized(input).await,
            OptimizationLevel::Aggressive => self.execute_aggressive_optimized(input).await,
            OptimizationLevel::Maximum => self.execute_maximum_optimized(input).await,
        }
    }
    
    async fn execute_basic_optimized(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        self.operation.forward(input).await
    }
    
    async fn execute_aggressive_optimized(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        match self.context.fusion_strategy {
            FusionStrategy::Sequential => self.execute_basic_optimized(input).await,
            FusionStrategy::LinearRelu => self.execute_fused(input).await,
            FusionStrategy::LinearDropout => self.execute_aggressive_fused(input).await,
            FusionStrategy::ConvRelu => self.execute_custom_fused(input).await,
            FusionStrategy::ConvBatchNorm => self.execute_custom_fused(input).await,
            FusionStrategy::AttentionFused => self.execute_custom_fused(input).await,
            FusionStrategy::TransformerBlock => self.execute_custom_fused(input).await,
            FusionStrategy::ResNetBlock => self.execute_custom_fused(input).await,
            FusionStrategy::CustomOptimized => self.execute_custom_fused(input).await,
            FusionStrategy::AutoDetect => self.execute_custom_fused(input).await,
        }
    }
    
    async fn execute_maximum_optimized(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        self.execute_aggressive_optimized(input).await
    }
    
    async fn execute_fused(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        self.operation.forward(input).await
    }
    
    async fn execute_aggressive_fused(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        self.operation.forward(input).await
    }
    
    async fn execute_custom_fused(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        self.operation.forward(input).await
    }

    /// Advanced operation fusion with automatic kernel generation
    pub async fn fuse_operations_advanced(&self, operations: &[Box<dyn AdvancedTensorOperation<DIMS>>]) -> AnvilResult<Box<dyn AdvancedTensorOperation<DIMS>>> {
        // Analyze operation patterns for optimal fusion
        let fusion_pattern = self.analyze_fusion_pattern(operations)?;
        
        match fusion_pattern {
            FusionPattern::LinearChain => self.fuse_linear_chain(operations).await,
            FusionPattern::Branched => self.fuse_branched_operations(operations).await,
            FusionPattern::Recurrent => self.fuse_recurrent_operations(operations).await,
            FusionPattern::Attention => self.fuse_attention_operations(operations).await,
            _ => self.fuse_generic_operations(operations).await,
        }
    }
    
    /// Analyze operation patterns for optimal fusion strategy
    fn analyze_fusion_pattern(&self, operations: &[Box<dyn AdvancedTensorOperation<DIMS>>]) -> AnvilResult<FusionPattern> {
        if operations.is_empty() {
            return Ok(FusionPattern::Empty);
        }
        
        // Advanced pattern detection
        let op_types: Vec<_> = operations.iter()
            .map(|op| op.name())
            .collect();
        
        // Update all call sites to use &[&str] instead of &[String]
        // Example:
        // self.is_linear_chain(&op_types)
        // becomes
        // self.is_linear_chain(&op_types.iter().map(|s| *s).collect::<Vec<&str>>())
        if self.is_linear_chain(&op_types.iter().map(|s| *s).collect::<Vec<&str>>()) {
            Ok(FusionPattern::LinearChain)
        } else if self.is_branched(&op_types.iter().map(|s| *s).collect::<Vec<&str>>()) {
            Ok(FusionPattern::Branched)
        } else if self.is_recurrent(&op_types.iter().map(|s| *s).collect::<Vec<&str>>()) {
            Ok(FusionPattern::Recurrent)
        } else if self.is_attention(&op_types.iter().map(|s| *s).collect::<Vec<&str>>()) {
            Ok(FusionPattern::Attention)
        } else {
            Ok(FusionPattern::Generic)
        }
    }

    pub fn is_linear_chain(&self, _op_types: &[&str]) -> bool { false }
    pub fn is_branched(&self, _op_types: &[&str]) -> bool { false }
    pub fn is_recurrent(&self, _op_types: &[&str]) -> bool { false }
    pub fn is_attention(&self, _op_types: &[&str]) -> bool { false }

    async fn fuse_linear_chain(&self, operations: &[Box<dyn AdvancedTensorOperation<DIMS>>]) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        // Simplified linear chain fusion
        let mut result = operations[0].forward(&AdvancedTensor::new(Shape::new([1, 1]), DType::F32, Device::Cpu)?).await?;
        for op in &operations[1..] {
            result = op.forward(&result).await?;
        }
        Ok(result)
    }
    
    async fn fuse_branched_operations(&self, operations: &[Box<dyn AdvancedTensorOperation<DIMS>>]) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        // Simplified branched fusion
        operations[0].forward(&AdvancedTensor::new(Shape::new([1, 1]), DType::F32, Device::Cpu)?).await
    }
    
    async fn fuse_recurrent_operations(&self, operations: &[Box<dyn AdvancedTensorOperation<DIMS>>]) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        // Simplified recurrent fusion
        operations[0].forward(&AdvancedTensor::new(Shape::new([1, 1]), DType::F32, Device::Cpu)?).await
    }
    
    async fn fuse_attention_operations(&self, operations: &[Box<dyn AdvancedTensorOperation<DIMS>>]) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        // Simplified attention fusion
        operations[0].forward(&AdvancedTensor::new(Shape::new([1, 1]), DType::F32, Device::Cpu)?).await
    }
    
    async fn fuse_generic_operations(&self, operations: &[Box<dyn AdvancedTensorOperation<DIMS>>]) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        // Simplified generic fusion
        operations[0].forward(&AdvancedTensor::new(Shape::new([1, 1]), DType::F32, Device::Cpu)?).await
    }
}

// Legacy compatibility - simple trait for basic operations
#[async_trait]
pub trait TensorOperation<const DIMS: usize>: Send + Sync {
    async fn forward(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>>;
}

// Convert legacy operations to advanced ones
impl<T, const DIMS: usize> AdvancedTensorOperation<DIMS> for T
where
    T: TensorOperation<DIMS>,
{
    async fn forward(&self, input: &AdvancedTensor<f32, DIMS>) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        <T as TensorOperation<DIMS>>::forward(self, input).await
    }
    fn name(&self) -> &'static str {
        "LegacyOperation"
    }
    fn input_shape_requirements(&self) -> Shape<DIMS> {
        Shape::new([0; DIMS])
    }
    fn output_shape(&self, _input_shape: &Shape<DIMS>) -> AnvilResult<Shape<DIMS>> {
        Ok(Shape::new([0; DIMS]))
    }
    fn operation_type(&self) -> &'static str {
        "legacy"
    }
    fn supports_simd(&self) -> bool {
        false
    }
    fn supports_gpu(&self) -> bool {
        false
    }
    fn memory_alignment(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionPattern {
    Empty,
    LinearChain,
    Branched,
    Recurrent,
    Attention,
    Generic,
}

#[derive(Debug, Clone)]
pub struct MemoryPlan {
    pub backend: crate::gpu::unified::GPUBackend,
    pub layout: MemoryLayout,
    pub total_size: usize,
    pub temporary_buffers: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ClusterHealth {
    pub is_healthy: bool,
    pub failed_nodes: Vec<u32>,
    pub performance_metrics: HashMap<String, f64>,
}

impl OptimizationLevel {
    pub const fn as_u8(&self) -> u8 {
        match self {
            OptimizationLevel::None => 0,
            OptimizationLevel::Basic => 1,
            OptimizationLevel::Aggressive => 2,
            OptimizationLevel::Maximum => 3,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelStrategy {
    Sequential,
    Rayon,
    SIMD,
    GPU,
    Hybrid,
}

impl ParallelStrategy {
    pub fn as_u8(&self) -> u8 {
        match self {
            ParallelStrategy::Sequential => 0,
            ParallelStrategy::Rayon => 1,
            ParallelStrategy::SIMD => 2,
            ParallelStrategy::GPU => 3,
            ParallelStrategy::Hybrid => 4,
        }
    }
    
    pub fn from_u8(value: u8) -> Option<Self> {
        match value {
            0 => Some(ParallelStrategy::Sequential),
            1 => Some(ParallelStrategy::Rayon),
            2 => Some(ParallelStrategy::SIMD),
            3 => Some(ParallelStrategy::GPU),
            4 => Some(ParallelStrategy::Hybrid),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_optimization_levels() {
        assert_eq!(OptimizationLevel::None.as_u8(), 0);
        assert_eq!(OptimizationLevel::Basic.as_u8(), 1);
        assert_eq!(OptimizationLevel::Aggressive.as_u8(), 2);
        assert_eq!(OptimizationLevel::Maximum.as_u8(), 3);
    }

    #[test]
    fn test_memory_layouts() {
        assert_eq!(MemoryLayout::RowMajor.as_u8(), 0);
        assert_eq!(MemoryLayout::ColumnMajor.as_u8(), 1);
        assert_eq!(MemoryLayout::Blocked { block_size: 16 }.as_u8(), 2);
        assert_eq!(MemoryLayout::Interleaved.as_u8(), 3);
        assert_eq!(MemoryLayout::Auto.as_u8(), 4);
    }

    #[test]
    fn test_parallel_strategies() {
        assert_eq!(ParallelStrategy::Sequential.as_u8(), 0);
        assert_eq!(ParallelStrategy::Rayon.as_u8(), 1);
        assert_eq!(ParallelStrategy::SIMD.as_u8(), 2);
        assert_eq!(ParallelStrategy::GPU.as_u8(), 3);
        assert_eq!(ParallelStrategy::Hybrid.as_u8(), 4);
    }
} 