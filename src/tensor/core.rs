//! Core tensor types and fundamental operations

use std::{
    fmt::{self, Debug, Display},
    marker::PhantomData,
};
use serde::{Serialize, Deserialize};

use crate::error::{AnvilError, AnvilResult};
use super::storage::AdvancedTensorStorage;
use super::devices::Device;
use super::ops::AdvancedAllocator;

/// Compile-time shape with const generics for type safety
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape<const DIMS: usize> {
    pub dims: [usize; DIMS],
}

impl<const DIMS: usize> Shape<DIMS> {
    pub fn new(dims: [usize; DIMS]) -> Self {
        Self { dims }
    }
    
    pub fn total_elements(&self) -> usize {
        self.dims.iter().product()
    }
    
    pub fn is_valid(&self) -> bool {
        self.dims.iter().all(|&d| d > 0)
    }
    
    pub fn as_slice(&self) -> &[usize] {
        &self.dims
    }
}

impl Shape<2> {
    pub fn rows(&self) -> usize { self.dims[0] }
    pub fn cols(&self) -> usize { self.dims[1] }
}

impl Shape<3> {
    pub fn depth(&self) -> usize { self.dims[0] }
    pub fn height(&self) -> usize { self.dims[1] }
    pub fn width(&self) -> usize { self.dims[2] }
}

impl Shape<4> {
    pub fn batch(&self) -> usize { self.dims[0] }
    pub fn channels(&self) -> usize { self.dims[1] }
    pub fn height(&self) -> usize { self.dims[2] }
    pub fn width(&self) -> usize { self.dims[3] }
}

impl<const DIMS: usize> Display for Shape<DIMS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[{}]", self.dims.iter()
            .map(|d| d.to_string())
            .collect::<Vec<_>>()
            .join(", "))
    }
}

/// Advanced data types with SIMD support and Python interop
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DType {
    F32,
    F64,
    F16,
    BF16,
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Bool,
    // Advanced types for quantization and specialized ops
    QI8,    // Quantized INT8
    QI4,    // Quantized INT4
    ComplexF32,
    ComplexF64,
}

impl DType {
    /// Get the size in bytes of this data type
    pub const fn size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I8 => 1,
            DType::I16 => 2,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::U16 => 2,
            DType::U32 => 4,
            DType::U64 => 8,
            DType::Bool => 1,
            DType::QI8 => 1,
            DType::QI4 => 1,
            DType::ComplexF32 => 8,
            DType::ComplexF64 => 16,
        }
    }

    /// Check if this is a floating point type
    pub const fn is_float(&self) -> bool {
        matches!(self, DType::F32 | DType::F64 | DType::F16 | DType::BF16)
    }

    /// Check if this is an integer type
    pub const fn is_int(&self) -> bool {
        matches!(self, DType::I8 | DType::I16 | DType::I32 | DType::I64 | 
                        DType::U8 | DType::U16 | DType::U32 | DType::U64)
    }

    /// Check if this is a quantized type
    pub const fn is_quantized(&self) -> bool {
        matches!(self, DType::QI8 | DType::QI4)
    }

    /// Check if this is a complex type
    pub const fn is_complex(&self) -> bool {
        matches!(self, DType::ComplexF32 | DType::ComplexF64)
    }

    /// Get optimal alignment for SIMD operations
    pub const fn simd_alignment(&self) -> usize {
        match self {
            DType::F32 => 16,  // SSE/NEON alignment
            DType::F64 => 16,
            DType::F16 => 16,
            DType::BF16 => 16,
            DType::I8 => 16,
            DType::I16 => 16,
            DType::I32 => 16,
            DType::I64 => 16,
            DType::U8 => 16,
            DType::U16 => 16,
            DType::U32 => 16,
            DType::U64 => 16,
            DType::Bool => 16,
            DType::QI8 => 16,
            DType::QI4 => 16,
            DType::ComplexF32 => 16,
            DType::ComplexF64 => 16,
        }
    }

    /// Convert to Python/NumPy dtype string (for PyO3 interop)
    pub fn to_numpy_dtype(&self) -> &'static str {
        match self {
            DType::F32 => "float32",
            DType::F64 => "float64",
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            DType::I8 => "int8",
            DType::I16 => "int16",
            DType::I32 => "int32",
            DType::I64 => "int64",
            DType::U8 => "uint8",
            DType::U16 => "uint16",
            DType::U32 => "uint32",
            DType::U64 => "uint64",
            DType::Bool => "bool",
            DType::QI8 => "int8",  // Quantized types map to base types
            DType::QI4 => "int8",
            DType::ComplexF32 => "complex64",
            DType::ComplexF64 => "complex128",
        }
    }
}

/// Advanced tensor with compile-time safety and zero-copy operations
#[derive(PartialEq)]
pub struct AdvancedTensor<D: Copy, const DIMS: usize> {
    storage: std::sync::Arc<AdvancedTensorStorage>,
    shape: Shape<DIMS>,
    dtype: DType,
    device: Device,
    _phantom: std::marker::PhantomData<D>,
}

impl<D: Copy, const DIMS: usize> AdvancedTensor<D, DIMS> {
    /// Create new tensor with advanced features
    pub fn new(shape: Shape<DIMS>, dtype: DType, device: Device) -> AnvilResult<Self> {
        let _allocator = std::sync::Arc::new(AdvancedAllocator::new(device, dtype.simd_alignment()));
        
        let storage = AdvancedTensorStorage::new(shape.total_elements() * dtype.size(), device);
        
        Ok(Self {
            storage: std::sync::Arc::new(storage),
            shape,
            dtype,
            device,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Create from vector with zero-copy when possible
    pub fn from_vec(data: Vec<f32>, shape: Shape<DIMS>) -> AnvilResult<Self> {
        if data.len() != shape.total_elements() {
            return Err(AnvilError::InvalidInput("Data length doesn't match shape".to_string()));
        }
        
        let storage = if data.len() == shape.total_elements() {
            let mut storage = AdvancedTensorStorage::new(shape.total_elements(), Device::Cpu);
            let dst_data = storage.as_slice_mut();
            // Copy data
            for (i, &value) in data.iter().enumerate() {
                let bytes = value.to_le_bytes();
                let start = i * 4;
                dst_data[start..start + 4].copy_from_slice(&bytes);
            }
            storage
        } else {
            return Err(AnvilError::InvalidInput("Data length doesn't match shape".to_string()));
        };
        
        let _allocator = std::sync::Arc::new(AdvancedAllocator::new(Device::Cpu, DType::F32.simd_alignment()));
        
        Ok(Self {
            storage: std::sync::Arc::new(storage),
            shape,
            dtype: DType::F32,
            device: Device::Cpu,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Get tensor shape
    pub fn shape(&self) -> Shape<DIMS> {
        self.shape.clone()
    }

    /// Get tensor data type
    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Get tensor device
    pub fn device(&self) -> Device {
        self.device
    }

    /// Get number of elements
    pub fn numel(&self) -> usize {
        self.shape.total_elements()
    }

    /// Get data as slice
    pub fn as_slice<T>(&self) -> &[T]
    where
        T: Copy + 'static,
    {
        // Convert from u8 slice to target type slice
        let u8_slice = unsafe { std::slice::from_raw_parts(self.storage.as_slice().as_ptr() as *const T, self.shape().total_elements()) };
        let size = std::mem::size_of::<T>();
        let len = u8_slice.len() / size;
        unsafe {
            std::slice::from_raw_parts(
                u8_slice.as_ptr() as *const T,
                len,
            )
        }
    }

    /// Get data as mutable slice
    pub fn as_slice_mut<T>(&mut self) -> &mut [T] {
        // This is a stub for now; real implementation would use bytemuck or similar
        if let Some(storage) = std::sync::Arc::get_mut(&mut self.storage) {
            unsafe { std::slice::from_raw_parts_mut(storage.as_slice_mut().as_mut_ptr() as *mut T, self.shape().total_elements()) }
        } else {
            &mut []
        }
    }
}

impl<D: Copy, const DIMS: usize> Clone for AdvancedTensor<D, DIMS> {
    fn clone(&self) -> Self {
        Self {
            storage: self.storage.clone(),
            shape: self.shape.clone(),
            dtype: self.dtype,
            device: self.device,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<D: Copy, const DIMS: usize> Debug for AdvancedTensor<D, DIMS> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AdvancedTensor")
            .field("shape", &self.shape)
            .field("dtype", &self.dtype())
            .field("device", &self.device())
            .field("size", &self.shape.total_elements())
            .finish()
    }
}

/// Zero-copy tensor view with advanced memory management
pub struct TensorView<const DIMS: usize> {
    storage: std::sync::Arc<AdvancedTensorStorage>,
    shape: Shape<DIMS>,
    dtype: DType,
    _phantom: PhantomData<Shape<DIMS>>,
}

impl<const DIMS: usize> TensorView<DIMS> {
    pub fn shape(&self) -> Shape<DIMS> {
        self.shape
    }
    
    pub fn dtype(&self) -> DType {
        self.dtype
    }
    
    pub fn as_slice<T>(&self) -> &[T]
    where
        T: Copy,
    {
        unsafe { std::slice::from_raw_parts(self.storage.as_slice().as_ptr() as *const T, self.shape().total_elements()) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_advanced_shape_creation() {
        let shape = Shape::new([2, 3, 4]);
        assert_eq!(shape.total_elements(), 24);
        assert!(shape.is_valid());
    }

    #[test]
    fn test_advanced_shape_validation() {
        let shape = Shape::new_validated([2, 3, 4]).unwrap();
        assert_eq!(shape.size_checked().unwrap(), 24);
        
        let invalid_shape = Shape::new_validated([2, 0, 4]);
        assert!(invalid_shape.is_err());
    }

    #[test]
    fn test_advanced_tensor_creation() {
        let shape = Shape::new([2, 2]);
        let tensor = AdvancedTensor::new(shape, DType::F32, Device::Cpu).unwrap();
        assert_eq!(tensor.shape().total_elements(), 4);
        assert_eq!(tensor.dtype(), DType::F32);
    }

    #[test]
    fn test_dtype_features() {
        assert!(DType::F32.is_float());
        assert!(DType::I32.is_int());
        assert!(DType::QI8.is_quantized());
        assert!(DType::ComplexF32.is_complex());
        assert_eq!(DType::F32.to_numpy_dtype(), "float32");
    }
} 