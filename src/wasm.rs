//! WebAssembly support for browser-based machine learning

use crate::{
    tensor::{Tensor, Shape, DType},
    error::{AnvilError, AnvilResult},
    nn::Module,
};

/// WebAssembly runtime environment
pub struct WasmRuntime {
    memory: Vec<u8>,
    heap_offset: usize,
}

impl WasmRuntime {
    pub fn new() -> Self {
        Self {
            memory: vec![0u8; 1024 * 1024], // 1MB initial memory
            heap_offset: 0,
        }
    }

    /// Allocate memory in the WASM heap
    pub fn allocate(&mut self, size: usize) -> usize {
        let ptr = self.heap_offset;
        self.heap_offset += size;
        
        // Ensure we have enough memory
        if self.heap_offset > self.memory.len() {
            self.memory.resize(self.heap_offset, 0);
        }
        
        ptr
    }

    /// Free memory (simplified - just track offset)
    pub fn free(&mut self, _ptr: usize) {
        // In a real implementation, we'd implement proper memory management
    }

    /// Get memory slice
    pub fn get_memory_slice(&self, ptr: usize, size: usize) -> &[u8] {
        &self.memory[ptr..ptr + size]
    }

    /// Get mutable memory slice
    pub fn get_memory_slice_mut(&mut self, ptr: usize, size: usize) -> &mut [u8] {
        &mut self.memory[ptr..ptr + size]
    }
}

/// WebAssembly tensor operations
pub struct WasmTensorOps;

impl WasmTensorOps {
    /// Matrix multiplication optimized for WASM
    pub fn matmul(a: &Tensor<2>, b: &Tensor<2>) -> AnvilResult<Tensor<2>> {
        // WASM-optimized matrix multiplication
        if !a.shape().matmul_compatible(&b.shape()) {
            return Err(AnvilError::InvalidShape(
                format!("Incompatible shapes for matrix multiplication: {:?} @ {:?}", 
                        a.shape(), b.shape())
            ));
        }

        let output_shape = a.shape().matmul_shape(&b.shape())
            .ok_or_else(|| AnvilError::InvalidShape("Cannot compute matmul output shape".to_string()))?;
        
        let mut output = Tensor::new(output_shape, a.dtype(), crate::tensor::Device::Cpu);
        
        // Perform matrix multiplication
        match a.dtype() {
            DType::F32 => {
                let a_data = a.as_slice::<f32>();
                let b_data = b.as_slice::<f32>();
                let mut c_data = output.as_slice_mut::<f32>();
                
                let m = a.shape().dims[0];
                let k = a.shape().dims[1];
                let n = b.shape().dims[1];
                
                // WASM-optimized matrix multiplication
                for i in 0..m {
                    for j in 0..n {
                        let mut sum = 0.0f32;
                        for l in 0..k {
                            sum += a_data[i * k + l] * b_data[l * n + j];
                        }
                        c_data[i * n + j] = sum;
                    }
                }
            }
            _ => return Err(AnvilError::UnsupportedOperation(
                format!("Matrix multiplication not supported for dtype: {:?}", a.dtype())
            )),
        }
        
        Ok(output)
    }

    /// Element-wise addition optimized for WASM
    pub fn add(a: &Tensor<2>, b: &Tensor<2>) -> AnvilResult<Tensor<2>> {
        if a.shape() != b.shape() {
            return Err(AnvilError::InvalidShape(
                format!("Shapes must match for addition: {:?} vs {:?}", 
                        a.shape(), b.shape())
            ));
        }

        let mut output = Tensor::new(a.shape(), a.dtype(), crate::tensor::Device::Cpu);
        
        match a.dtype() {
            DType::F32 => {
                let a_data = a.as_slice::<f32>();
                let b_data = b.as_slice::<f32>();
                let mut c_data = output.as_slice_mut::<f32>();
                
                for i in 0..a_data.len() {
                    c_data[i] = a_data[i] + b_data[i];
                }
            }
            _ => return Err(AnvilError::UnsupportedOperation(
                format!("Addition not supported for dtype: {:?}", a.dtype())
            )),
        }
        
        Ok(output)
    }
}

/// WebAssembly model loader
pub struct WasmModelLoader;

impl WasmModelLoader {
    pub fn new() -> Self {
        Self
    }

    /// Load a model from WASM memory
    pub fn load_model(&self, data: &[u8]) -> AnvilResult<Box<dyn Module>> {
        // Parse model data and create module
        // This is a simplified implementation
        tracing::info!("Loading model from WASM memory ({} bytes)", data.len());
        
        // For now, return a simple linear layer
        let linear = crate::nn::Linear::new(10, 1, crate::tensor::Device::Cpu);
        Ok(Box::new(linear))
    }

    /// Save a model to WASM memory
    pub fn save_model(&self, model: &dyn Module) -> AnvilResult<Vec<u8>> {
        // Serialize model to bytes
        tracing::info!("Saving model to WASM memory");
        
        // Simplified serialization
        let mut data = Vec::new();
        data.extend_from_slice(b"ANVIL_MODEL");
        data.extend_from_slice(&(model.parameters().len() as u32).to_le_bytes());
        
        Ok(data)
    }
}

/// WebAssembly inference engine
pub struct WasmInferenceEngine {
    runtime: WasmRuntime,
    model: Option<Box<dyn Module>>,
}

impl WasmInferenceEngine {
    pub fn new() -> Self {
        Self {
            runtime: WasmRuntime::new(),
            model: None,
        }
    }

    /// Load a model
    pub fn load_model(&mut self, model_data: &[u8]) -> AnvilResult<()> {
        let loader = WasmModelLoader::new();
        self.model = Some(loader.load_model(model_data)?);
        Ok(())
    }

    /// Run inference
    pub async fn infer(&self, input: &Tensor<2>) -> AnvilResult<Tensor<2>> {
        if let Some(ref model) = self.model {
            model.forward(input).await
        } else {
            Err(AnvilError::InvalidState("No model loaded".to_string()))
        }
    }

    /// Get memory usage statistics
    pub fn memory_stats(&self) -> WasmMemoryStats {
        WasmMemoryStats {
            total_memory: self.runtime.memory.len(),
            heap_used: self.runtime.heap_offset,
            heap_available: self.runtime.memory.len() - self.runtime.heap_offset,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WasmMemoryStats {
    pub total_memory: usize,
    pub heap_used: usize,
    pub heap_available: usize,
}

/// WebAssembly utilities
pub mod utils {
    use super::*;

    /// Convert JavaScript ArrayBuffer to Rust Vec<u8>
    pub fn array_buffer_to_vec(buffer: &[u8]) -> Vec<u8> {
        buffer.to_vec()
    }

    /// Convert Rust Vec<u8> to JavaScript ArrayBuffer
    pub fn vec_to_array_buffer(data: &[u8]) -> Vec<u8> {
        data.to_vec()
    }

    /// Create a tensor from JavaScript data
    pub fn tensor_from_js_data(data: &[f32], shape: &[usize]) -> AnvilResult<Tensor<2>> {
        if shape.len() != 2 {
            return Err(AnvilError::InvalidShape(
                "Only 2D tensors supported for WASM".to_string()
            ));
        }

        let tensor_shape = Shape::new([shape[0], shape[1]]);
        let mut tensor = Tensor::new(tensor_shape, DType::F32, crate::tensor::Device::Cpu);
        
        // Copy data to tensor
        let mut tensor_data = tensor.as_slice_mut::<f32>();
        tensor_data.copy_from_slice(data);
        
        Ok(tensor)
    }

    /// Convert tensor to JavaScript data
    pub fn tensor_to_js_data(tensor: &Tensor<2>) -> Vec<f32> {
        tensor.as_slice::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};

    #[test]
    fn test_wasm_runtime() {
        let mut runtime = WasmRuntime::new();
        let ptr = runtime.allocate(1024);
        assert_eq!(ptr, 0);
        
        let ptr2 = runtime.allocate(512);
        assert_eq!(ptr2, 1024);
    }

    #[test]
    fn test_wasm_tensor_ops() {
        let a = Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu);
        let b = Tensor::new(Shape::new([3, 2]), DType::F32, Device::Cpu);
        
        let result = WasmTensorOps::matmul(&a, &b);
        assert!(result.is_ok());
    }

    #[test]
    fn test_wasm_inference_engine() {
        let mut engine = WasmInferenceEngine::new();
        let model_data = b"ANVIL_MODEL";
        
        let result = engine.load_model(model_data);
        assert!(result.is_ok());
        
        let stats = engine.memory_stats();
        assert!(stats.total_memory > 0);
    }
} 