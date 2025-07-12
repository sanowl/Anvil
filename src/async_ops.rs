//! Asynchronous tensor operations for non-blocking execution

use std::sync::Arc;
use tokio::sync::mpsc;
use crate::{
    tensor::{Tensor, Shape, DType},
    error::{AnvilError, AnvilResult},
    ops::TensorOperation,
};

/// Asynchronous operation executor
pub struct AsyncOpExecutor {
    tx: mpsc::Sender<AsyncOp>,
    rx: mpsc::Receiver<AsyncOpResult>,
}

#[derive(Debug)]
enum AsyncOp {
    MatMul {
        a: Arc<Tensor<2>>,
        b: Arc<Tensor<2>>,
        result_tx: tokio::sync::oneshot::Sender<AnvilResult<Tensor<2>>>,
    },
    Add {
        a: Arc<Tensor<2>>,
        b: Arc<Tensor<2>>,
        result_tx: tokio::sync::oneshot::Sender<AnvilResult<Tensor<2>>>,
    },
    Conv2d {
        input: Arc<Tensor<4>>,
        weight: Arc<Tensor<4>>,
        result_tx: tokio::sync::oneshot::Sender<AnvilResult<Tensor<4>>>,
    },
}

#[derive(Debug)]
enum AsyncOpResult {
    MatMul(AnvilResult<Tensor<2>>),
    Add(AnvilResult<Tensor<2>>),
    Conv2d(AnvilResult<Tensor<4>>),
}

impl AsyncOpExecutor {
    pub fn new() -> Self {
        let (tx, mut rx) = mpsc::channel(100);
        let (result_tx, result_rx) = mpsc::channel(100);
        
        // Spawn worker thread
        tokio::spawn(async move {
            while let Some(op) = rx.recv().await {
                let result = match op {
                    AsyncOp::MatMul { a, b, result_tx } => {
                        let result = a.matmul(&b);
                        let _ = result_tx.send(result);
                    }
                    AsyncOp::Add { a, b, result_tx } => {
                        let result = a.add(&b);
                        let _ = result_tx.send(result);
                    }
                    AsyncOp::Conv2d { input, weight, result_tx } => {
                        // Simplified convolution for now
                        let result = Err(AnvilError::UnsupportedOperation(
                            "Async convolution not yet implemented".to_string()
                        ));
                        let _ = result_tx.send(result);
                    }
                };
            }
        });

        Self { tx, rx: result_rx }
    }

    /// Asynchronously perform matrix multiplication
    pub async fn matmul(&self, a: &Tensor<2>, b: &Tensor<2>) -> AnvilResult<Tensor<2>> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        
        let op = AsyncOp::MatMul {
            a: Arc::new(a.clone()),
            b: Arc::new(b.clone()),
            result_tx,
        };

        self.tx.send(op).await
            .map_err(|_| AnvilError::DeviceError("Async executor channel closed".to_string()))?;

        result_rx.await
            .map_err(|_| AnvilError::DeviceError("Result channel closed".to_string()))?
    }

    /// Asynchronously perform element-wise addition
    pub async fn add(&self, a: &Tensor<2>, b: &Tensor<2>) -> AnvilResult<Tensor<2>> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        
        let op = AsyncOp::Add {
            a: Arc::new(a.clone()),
            b: Arc::new(b.clone()),
            result_tx,
        };

        self.tx.send(op).await
            .map_err(|_| AnvilError::DeviceError("Async executor channel closed".to_string()))?;

        result_rx.await
            .map_err(|_| AnvilError::DeviceError("Result channel closed".to_string()))?
    }

    /// Asynchronously perform 2D convolution
    pub async fn conv2d(&self, input: &Tensor<4>, weight: &Tensor<4>) -> AnvilResult<Tensor<4>> {
        let (result_tx, result_rx) = tokio::sync::oneshot::channel();
        
        let op = AsyncOp::Conv2d {
            input: Arc::new(input.clone()),
            weight: Arc::new(weight.clone()),
            result_tx,
        };

        self.tx.send(op).await
            .map_err(|_| AnvilError::DeviceError("Async executor channel closed".to_string()))?;

        result_rx.await
            .map_err(|_| AnvilError::DeviceError("Result channel closed".to_string()))?
    }
}

/// Asynchronous operation pipeline
pub struct AsyncPipeline {
    operations: Vec<Box<dyn TensorOperation + Send + Sync>>,
    executor: AsyncOpExecutor,
}

impl AsyncPipeline {
    pub fn new() -> Self {
        Self {
            operations: Vec::new(),
            executor: AsyncOpExecutor::new(),
        }
    }

    /// Add an operation to the pipeline
    pub fn add_operation(&mut self, op: Box<dyn TensorOperation + Send + Sync>) {
        self.operations.push(op);
    }

    /// Execute the pipeline asynchronously
    pub async fn execute(&self, input: &Tensor<2>) -> AnvilResult<Tensor<2>> {
        let mut current = input.clone();
        
        for op in &self.operations {
            current = op.execute(&current).await?;
        }
        
        Ok(current)
    }

    /// Execute the pipeline with progress callback
    pub async fn execute_with_progress<F>(
        &self,
        input: &Tensor<2>,
        mut progress_callback: F,
    ) -> AnvilResult<Tensor<2>>
    where
        F: FnMut(f32) + Send,
    {
        let mut current = input.clone();
        let total_ops = self.operations.len();
        
        for (i, op) in self.operations.iter().enumerate() {
            current = op.execute(&current).await?;
            let progress = (i + 1) as f32 / total_ops as f32;
            progress_callback(progress);
        }
        
        Ok(current)
    }
}

/// Asynchronous batch processor
pub struct AsyncBatchProcessor {
    batch_size: usize,
    executor: AsyncOpExecutor,
}

impl AsyncBatchProcessor {
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            executor: AsyncOpExecutor::new(),
        }
    }

    /// Process a batch of tensors asynchronously
    pub async fn process_batch(
        &self,
        inputs: Vec<Tensor<2>>,
    ) -> AnvilResult<Vec<Tensor<2>>> {
        let mut results = Vec::new();
        let mut futures = Vec::new();

        // Split into batches
        for chunk in inputs.chunks(self.batch_size) {
            let chunk = chunk.to_vec(); // Clone to own the data
            let mut batch_futures = Vec::new();
            
            for input in chunk {
                // For now, just apply a simple operation (identity)
                let future = tokio::spawn(async move {
                    Ok::<Tensor<2>, AnvilError>(input.clone())
                });
                batch_futures.push(future);
            }
            
            futures.push(batch_futures);
        }

        // Wait for all batches to complete
        for batch_futures in futures {
            for future in batch_futures {
                let result = future.await
                    .map_err(|_| AnvilError::DeviceError("Task join failed".to_string()))??;
                results.push(result);
            }
        }

        Ok(results)
    }
}

/// Global async executor
lazy_static::lazy_static! {
    pub static ref ASYNC_EXECUTOR: Arc<AsyncOpExecutor> = Arc::new(AsyncOpExecutor::new());
}

/// Convenience function for async matrix multiplication
pub async fn async_matmul(a: &Tensor<2>, b: &Tensor<2>) -> AnvilResult<Tensor<2>> {
    ASYNC_EXECUTOR.matmul(a, b).await
}

/// Convenience function for async addition
pub async fn async_add(a: &Tensor<2>, b: &Tensor<2>) -> AnvilResult<Tensor<2>> {
    ASYNC_EXECUTOR.add(a, b).await
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};

    #[tokio::test]
    async fn test_async_matmul() {
        let a = Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu);
        let b = Tensor::new(Shape::new([3, 2]), DType::F32, Device::Cpu);
        
        let result = async_matmul(&a, &b).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_async_pipeline() {
        let mut pipeline = AsyncPipeline::new();
        let input = Tensor::new(Shape::new([2, 2]), DType::F32, Device::Cpu);
        
        let result = pipeline.execute(&input).await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_batch_processor() {
        let processor = AsyncBatchProcessor::new(2);
        let inputs = vec![
            Tensor::new(Shape::new([2, 2]), DType::F32, Device::Cpu),
            Tensor::new(Shape::new([2, 2]), DType::F32, Device::Cpu),
        ];
        
        let results = processor.process_batch(inputs).await;
        assert!(results.is_ok());
        assert_eq!(results.unwrap().len(), 2);
    }
} 