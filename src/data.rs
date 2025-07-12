use std::{
    collections::VecDeque,
    sync::{Arc, Mutex},
    path::PathBuf,
};
use async_trait::async_trait;
use tokio::sync::mpsc;
use tokio::sync::Mutex as TokioMutex;
use serde::{Serialize, Deserialize};

use crate::{
    tensor::{Tensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
    memory::get_memory_manager,
};

/// Base trait for all datasets
#[async_trait]
pub trait Dataset: Send + Sync {
    /// Get the length of the dataset
    fn len(&self) -> usize;
    
    /// Check if the dataset is empty
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    
    /// Get an item at the specified index
    async fn get(&self, index: usize) -> AnvilResult<DataItem>;
    
    /// Get multiple items
    async fn get_batch(&self, indices: &[usize]) -> AnvilResult<Vec<DataItem>>;
    
    /// Get the shape of input data
    fn input_shape(&self) -> Shape<2>;
    
    /// Get the shape of target data
    fn target_shape(&self) -> Shape<2>;
    
    /// Get the data type
    fn dtype(&self) -> DType;
    
    /// Get the device
    fn device(&self) -> Device;
}

/// Data item containing input and target
#[derive(Debug, Clone)]
pub struct DataItem {
    pub input: Tensor<2>,
    pub target: Tensor<2>,
    pub metadata: Option<serde_json::Value>,
}

impl DataItem {
    pub fn new(input: Tensor<2>, target: Tensor<2>) -> Self {
        Self {
            input,
            target,
            metadata: None,
        }
    }
    
    pub fn with_metadata(mut self, metadata: serde_json::Value) -> Self {
        self.metadata = Some(metadata);
        self
    }

    /// Get the data type of this item
    pub fn dtype(&self) -> DType {
        self.input.dtype()
    }

    /// Get the device of this item
    pub fn device(&self) -> Device {
        self.input.device()
    }
}

/// Streaming dataset for infinite data processing
pub struct StreamingDataset {
    source: Arc<TokioMutex<Box<dyn DataSource>>>,
    buffer_size: usize,
    buffer: Arc<std::sync::Mutex<VecDeque<DataItem>>>,
    device: Device,
    dtype: DType,
}

#[async_trait]
pub trait DataSource: Send + Sync {
    async fn next(&mut self) -> AnvilResult<Option<DataItem>>;
    async fn reset(&mut self) -> AnvilResult<()>;
}

impl StreamingDataset {
    pub fn new(source: Box<dyn DataSource>, buffer_size: usize, device: Device) -> Self {
        Self {
            source: Arc::new(TokioMutex::new(source)),
            buffer_size,
            buffer: Arc::new(std::sync::Mutex::new(VecDeque::new())),
            device,
            dtype: DType::F32,
        }
    }
    
    pub async fn fill_buffer(&self) -> AnvilResult<()> {
        let mut source = self.source.lock().await;
        while self.buffer.lock().unwrap().len() < self.buffer_size {
            if let Some(item) = source.next().await? {
                self.buffer.lock().unwrap().push_back(item);
            } else {
                source.reset().await?;
            }
        }
        Ok(())
    }
    
    pub async fn get_next(&self) -> AnvilResult<Option<DataItem>> {
        let mut source = self.source.lock().await;
        if let Ok(Some(item)) = source.next().await {
            return Ok(Some(item));
        }
        Ok(None)
    }
}

#[async_trait]
impl Dataset for StreamingDataset {
    fn len(&self) -> usize {
        // Streaming datasets have infinite length
        usize::MAX
    }
    
    async fn get(&self, _index: usize) -> AnvilResult<DataItem> {
        self.get_next().await?.ok_or_else(|| {
            AnvilError::OperationError {
                operation: "get".to_string(),
                message: "No more data available".to_string(),
            }
        })
    }
    
    async fn get_batch(&self, indices: &[usize]) -> AnvilResult<Vec<DataItem>> {
        let mut items = Vec::new();
        for _ in 0..indices.len() {
            if let Some(item) = self.get_next().await? {
                items.push(item);
            } else {
                break;
            }
        }
        Ok(items)
    }
    
    fn input_shape(&self) -> Shape<2> {
        // This would be determined by the actual data
        Shape::new([0, 0])
    }
    
    fn target_shape(&self) -> Shape<2> {
        // This would be determined by the actual data
        Shape::new([0, 0])
    }
    
    fn dtype(&self) -> DType {
        self.dtype
    }
    
    fn device(&self) -> Device {
        self.device
    }
}

/// Dynamic batch optimizer for variable-length sequences
pub struct DynamicBatchOptimizer {
    max_batch_size: usize,
    max_sequence_length: usize,
    memory_manager: Arc<crate::memory::MemoryManager>,
}

impl DynamicBatchOptimizer {
    pub fn new(max_batch_size: usize, max_sequence_length: usize) -> Self {
        Self {
            max_batch_size,
            max_sequence_length,
            memory_manager: get_memory_manager(),
        }
    }
    
    pub fn optimize_batch(&self, items: &[DataItem]) -> Vec<Vec<DataItem>> {
        let mut batches = Vec::new();
        let mut current_batch = Vec::new();
        let mut current_max_length = 0;
        
        for item in items {
            let sequence_length = item.input.shape().dims[0];
            
            // Check if adding this item would exceed limits
            if current_batch.len() >= self.max_batch_size ||
               current_max_length.max(sequence_length) > self.max_sequence_length {
                // Start a new batch
                if !current_batch.is_empty() {
                    batches.push(current_batch);
                }
                current_batch = vec![item.clone()];
                current_max_length = sequence_length;
            } else {
                // Add to current batch
                current_batch.push(item.clone());
                current_max_length = current_max_length.max(sequence_length);
            }
        }
        
        // Add the last batch
        if !current_batch.is_empty() {
            batches.push(current_batch);
        }
        
        batches
    }
    
    pub fn pad_batch(&self, batch: &[DataItem]) -> AnvilResult<(Tensor<2>, Tensor<2>)> {
        if batch.is_empty() {
            return Err(AnvilError::OperationError {
                operation: "pad_batch".to_string(),
                message: "Empty batch".to_string(),
            });
        }
        
        // Find the maximum sequence length in the batch
        let max_length = batch.iter()
            .map(|item| item.input.shape().dims[0])
            .max()
            .unwrap();
        
        let batch_size = batch.len();
        let input_dim = batch[0].input.shape().dims[1];
        let target_dim = batch[0].target.shape().dims[1];
        
        // Create padded tensors
        let mut padded_input = Tensor::new(
            Shape::new([batch_size * max_length, input_dim]),
            batch[0].dtype(),
            batch[0].device(),
        );
        
        let mut padded_target = Tensor::new(
            Shape::new([batch_size * max_length, target_dim]),
            batch[0].dtype(),
            batch[0].device(),
        );
        
        // Fill with padding
        for (i, item) in batch.iter().enumerate() {
            let item_length = item.input.shape().dims[0];
            
            // Copy input data
            for j in 0..item_length {
                let src_idx = j * input_dim;
                let dst_idx = (i * max_length + j) * input_dim;
                
                // Copy input values
                for k in 0..input_dim {
                    // This would copy the actual values
                    let src_idx = j * input_dim + k;
                    let dst_idx = (i * max_length + j) * input_dim + k;
                    
                    // Copy based on dtype
                    match item.input.dtype() {
                        DType::F32 => {
                            let src_data = item.input.as_slice::<f32>();
                            let mut dst_data = padded_input.as_slice_mut::<f32>();
                            dst_data[dst_idx] = src_data[src_idx];
                        }
                        DType::F64 => {
                            let src_data = item.input.as_slice::<f64>();
                            let mut dst_data = padded_input.as_slice_mut::<f64>();
                            dst_data[dst_idx] = src_data[src_idx];
                        }
                        _ => {
                            // For other dtypes, copy as bytes
                            let src_data = item.input.as_slice::<u8>();
                            let mut dst_data = padded_input.as_slice_mut::<u8>();
                            let elem_size = item.input.dtype().size();
                            let src_start = src_idx * elem_size;
                            let dst_start = dst_idx * elem_size;
                            dst_data[dst_start..dst_start + elem_size]
                                .copy_from_slice(&src_data[src_start..src_start + elem_size]);
                        }
                    }
                }
            }
            
            // Copy target data
            for j in 0..item_length {
                let src_idx = j * target_dim;
                let dst_idx = (i * max_length + j) * target_dim;
                
                // Copy target values
                for k in 0..target_dim {
                    // This would copy the actual values
                    let src_idx = j * target_dim + k;
                    let dst_idx = (i * max_length + j) * target_dim + k;
                    
                    // Copy based on dtype
                    match item.target.dtype() {
                        DType::F32 => {
                            let src_data = item.target.as_slice::<f32>();
                            let mut dst_data = padded_target.as_slice_mut::<f32>();
                            dst_data[dst_idx] = src_data[src_idx];
                        }
                        DType::F64 => {
                            let src_data = item.target.as_slice::<f64>();
                            let mut dst_data = padded_target.as_slice_mut::<f64>();
                            dst_data[dst_idx] = src_data[src_idx];
                        }
                        _ => {
                            // For other dtypes, copy as bytes
                            let src_data = item.target.as_slice::<u8>();
                            let mut dst_data = padded_target.as_slice_mut::<u8>();
                            let elem_size = item.target.dtype().size();
                            let src_start = src_idx * elem_size;
                            let dst_start = dst_idx * elem_size;
                            dst_data[dst_start..dst_start + elem_size]
                                .copy_from_slice(&src_data[src_start..src_start + elem_size]);
                        }
                    }
                }
            }
        }
        
        Ok((padded_input, padded_target))
    }
}

/// Data loader with automatic batching and prefetching
pub struct DataLoader {
    dataset: Arc<dyn Dataset>,
    batch_size: usize,
    shuffle: bool,
    num_workers: usize,
    prefetch_factor: usize,
    dynamic_batching: bool,
    batch_optimizer: Option<DynamicBatchOptimizer>,
    memory_manager: Arc<crate::memory::MemoryManager>,
}

impl DataLoader {
    pub fn new(dataset: Arc<dyn Dataset>, batch_size: usize) -> Self {
        Self {
            dataset,
            batch_size,
            shuffle: false,
            num_workers: 0,
            prefetch_factor: 2,
            dynamic_batching: false,
            batch_optimizer: None,
            memory_manager: get_memory_manager(),
        }
    }
    
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }
    
    pub fn with_workers(mut self, num_workers: usize) -> Self {
        self.num_workers = num_workers;
        self
    }
    
    pub fn with_prefetch(mut self, prefetch_factor: usize) -> Self {
        self.prefetch_factor = prefetch_factor;
        self
    }
    
    pub fn with_dynamic_batching(mut self, dynamic_batching: bool) -> Self {
        self.dynamic_batching = dynamic_batching;
        if dynamic_batching {
            self.batch_optimizer = Some(DynamicBatchOptimizer::new(
                self.batch_size,
                1024, // Default max sequence length
            ));
        }
        self
    }
    
    pub async fn iter(self: &Arc<Self>) -> DataLoaderIter {
        DataLoaderIter::new(Arc::clone(self)).await
    }
    
    pub async fn get_batch(&self, indices: &[usize]) -> AnvilResult<Vec<DataItem>> {
        if self.num_workers > 0 {
            self.get_batch_parallel(indices).await
        } else {
            self.dataset.get_batch(indices).await
        }
    }
    
    async fn get_batch_parallel(&self, indices: &[usize]) -> AnvilResult<Vec<DataItem>> {
        let chunk_size = (indices.len() + self.num_workers - 1) / self.num_workers;
        let mut handles = Vec::new();
        
        for chunk in indices.chunks(chunk_size) {
            let dataset = self.dataset.clone();
            let chunk = chunk.to_vec();
            
            let handle = tokio::spawn(async move {
                dataset.get_batch(&chunk).await
            });
            
            handles.push(handle);
        }
        
        let mut results = Vec::new();
        for handle in handles {
            let chunk_result = handle.await.map_err(|e| {
                AnvilError::OperationError {
                    operation: "parallel_batch".to_string(),
                    message: format!("Worker task failed: {}", e),
                }
            })??;
            
            results.extend(chunk_result);
        }
        
        Ok(results)
    }
}

/// Iterator for data loader
pub struct DataLoaderIter {
    loader: Arc<DataLoader>,
    current_index: usize,
    indices: Vec<usize>,
    prefetch_queue: mpsc::Receiver<Vec<DataItem>>,
    prefetch_sender: mpsc::Sender<Vec<DataItem>>,
}

impl DataLoaderIter {
    pub async fn new(loader: Arc<DataLoader>) -> Self {
        let dataset_len = loader.dataset.len();
        let mut indices: Vec<usize> = (0..dataset_len).collect();
        
        if loader.shuffle {
            // Shuffle indices
            use rand::seq::SliceRandom;
            let mut rng = rand::thread_rng();
            indices.shuffle(&mut rng);
        }
        
        let (tx, rx) = mpsc::channel(loader.prefetch_factor);
        
        let loader_for_task = Arc::clone(&loader);
        let indices_for_task = indices.clone();
        let sender = tx.clone();
        
        // Start prefetching
        tokio::spawn(async move {
            let mut i = 0;
            while i < indices_for_task.len() {
                let batch_indices: Vec<usize> = indices_for_task[i..]
                    .iter()
                    .take(loader_for_task.batch_size)
                    .cloned()
                    .collect();
                
                if let Ok(batch) = loader_for_task.get_batch(&batch_indices).await {
                    if sender.send(batch).await.is_err() {
                        break;
                    }
                }
                
                i += loader_for_task.batch_size;
            }
        });
        
        Self {
            loader: loader,
            current_index: 0,
            indices,
            prefetch_queue: rx,
            prefetch_sender: tx,
        }
    }
}

impl Iterator for DataLoaderIter {
    type Item = Vec<DataItem>;
    
    fn next(&mut self) -> Option<Self::Item> {
        // This would be implemented as an async iterator in a real implementation
        // For now, we'll use a blocking approach with tokio runtime
        if self.current_index >= self.indices.len() {
            return None;
        }
        
        // Get the next batch from the prefetch queue
        match self.prefetch_queue.try_recv() {
            Ok(batch) => {
                self.current_index += self.loader.batch_size;
                Some(batch)
            }
            Err(_) => {
                // No data available, return None
                None
            }
        }
    }
}

/// File-based dataset
pub struct FileDataset {
    file_path: PathBuf,
    device: Device,
    dtype: DType,
    input_shape: Shape<2>,
    target_shape: Shape<2>,
    length: usize,
}

impl FileDataset {
    pub fn new(file_path: PathBuf, device: Device) -> AnvilResult<Self> {
        // In a real implementation, this would read the file metadata
        let input_shape = Shape::new([0, 0]); // Would be determined from file
        let target_shape = Shape::new([0, 0]); // Would be determined from file
        let length = 0; // Would be determined from file
        
        Ok(Self {
            file_path,
            device,
            dtype: DType::F32,
            input_shape,
            target_shape,
            length,
        })
    }
}

#[async_trait]
impl Dataset for FileDataset {
    fn len(&self) -> usize {
        self.length
    }
    
    async fn get(&self, index: usize) -> AnvilResult<DataItem> {
        if index >= self.length {
            return Err(AnvilError::OperationError {
                operation: "get".to_string(),
                message: format!("Index {} out of bounds", index),
            });
        }
        
        // In a real implementation, this would read from the file
        let input = Tensor::new(self.input_shape, self.dtype, self.device);
        let target = Tensor::new(self.target_shape, self.dtype, self.device);
        
        Ok(DataItem::new(input, target))
    }
    
    async fn get_batch(&self, indices: &[usize]) -> AnvilResult<Vec<DataItem>> {
        let mut items = Vec::new();
        for &index in indices {
            items.push(self.get(index).await?);
        }
        Ok(items)
    }
    
    fn input_shape(&self) -> Shape<2> {
        self.input_shape
    }
    
    fn target_shape(&self) -> Shape<2> {
        self.target_shape
    }
    
    fn dtype(&self) -> DType {
        self.dtype
    }
    
    fn device(&self) -> Device {
        self.device
    }
}

/// In-memory dataset
pub struct InMemoryDataset {
    data: Vec<DataItem>,
    device: Device,
    dtype: DType,
}

impl InMemoryDataset {
    pub fn new(data: Vec<DataItem>, device: Device) -> Self {
        let dtype = if !data.is_empty() {
            data[0].dtype()
        } else {
            DType::F32
        };
        
        Self {
            data,
            device,
            dtype,
        }
    }
    
    pub fn from_tensors(inputs: Vec<Tensor<2>>, targets: Vec<Tensor<2>>, device: Device) -> AnvilResult<Self> {
        if inputs.len() != targets.len() {
            return Err(AnvilError::OperationError {
                operation: "from_tensors".to_string(),
                message: "Input and target counts must match".to_string(),
            });
        }
        
        let data: Vec<DataItem> = inputs.into_iter()
            .zip(targets.into_iter())
            .map(|(input, target)| DataItem::new(input, target))
            .collect();
        
        Ok(Self::new(data, device))
    }
}

#[async_trait]
impl Dataset for InMemoryDataset {
    fn len(&self) -> usize {
        self.data.len()
    }
    
    async fn get(&self, index: usize) -> AnvilResult<DataItem> {
        if index >= self.data.len() {
            return Err(AnvilError::OperationError {
                operation: "get".to_string(),
                message: format!("Index {} out of bounds", index),
            });
        }
        
        Ok(self.data[index].clone())
    }
    
    async fn get_batch(&self, indices: &[usize]) -> AnvilResult<Vec<DataItem>> {
        let mut items = Vec::new();
        for &index in indices {
            items.push(self.get(index).await?);
        }
        Ok(items)
    }
    
    fn input_shape(&self) -> Shape<2> {
        if !self.data.is_empty() {
            self.data[0].input.shape()
        } else {
            Shape::new([0, 0])
        }
    }
    
    fn target_shape(&self) -> Shape<2> {
        if !self.data.is_empty() {
            self.data[0].target.shape()
        } else {
            Shape::new([0, 0])
        }
    }
    
    fn dtype(&self) -> DType {
        self.dtype
    }
    
    fn device(&self) -> Device {
        self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_in_memory_dataset() {
        let inputs = vec![
            Tensor::new(Shape::new([1, 784]), DType::F32, Device::Cpu),
            Tensor::new(Shape::new([1, 784]), DType::F32, Device::Cpu),
        ];
        let targets = vec![
            Tensor::new(Shape::new([1, 10]), DType::F32, Device::Cpu),
            Tensor::new(Shape::new([1, 10]), DType::F32, Device::Cpu),
        ];
        
        let dataset = InMemoryDataset::from_tensors(inputs, targets, Device::Cpu).unwrap();
        assert_eq!(dataset.len(), 2);
        
        let item = dataset.get(0).await.unwrap();
        assert_eq!(item.input.shape().dims, [1, 784]);
        assert_eq!(item.target.shape().dims, [1, 10]);
    }

    #[tokio::test]
    async fn test_data_loader() {
        let inputs = vec![
            Tensor::new(Shape::new([1, 784]), DType::F32, Device::Cpu),
            Tensor::new(Shape::new([1, 784]), DType::F32, Device::Cpu),
        ];
        let targets = vec![
            Tensor::new(Shape::new([1, 10]), DType::F32, Device::Cpu),
            Tensor::new(Shape::new([1, 10]), DType::F32, Device::Cpu),
        ];
        
        let dataset = Arc::new(InMemoryDataset::from_tensors(inputs, targets, Device::Cpu).unwrap());
        let loader = DataLoader::new(dataset, 2)
            .with_shuffle(true)
            .with_workers(2);
        
        let batch = loader.get_batch(&[0, 1]).await.unwrap();
        assert_eq!(batch.len(), 2);
    }

    #[tokio::test]
    async fn test_dynamic_batch_optimizer() {
        let optimizer = DynamicBatchOptimizer::new(4, 100);
        
        let items = vec![
            DataItem::new(
                Tensor::new(Shape::new([10, 256]), DType::F32, Device::Cpu),
                Tensor::new(Shape::new([10, 10]), DType::F32, Device::Cpu),
            ),
            DataItem::new(
                Tensor::new(Shape::new([20, 256]), DType::F32, Device::Cpu),
                Tensor::new(Shape::new([20, 10]), DType::F32, Device::Cpu),
            ),
        ];
        
        let batches = optimizer.optimize_batch(&items);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 2);
    }
} 