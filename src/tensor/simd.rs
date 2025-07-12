//! SIMD-accelerated tensor operations

use rayon::prelude::*;
use crate::error::{AnvilError, AnvilResult};
use super::core::{AdvancedTensor, Shape, DType};

// SIMD support
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

impl<T, const DIMS: usize> AdvancedTensor<T, DIMS> 
where
    T: Copy + Clone + Default + Send + Sync,
{
    /// SIMD-accelerated element-wise addition
    pub fn add_simd(&self, other: &Self) -> AnvilResult<Self> {
        if self.shape() != other.shape() {
            return Err(AnvilError::InvalidInput("Shape mismatch".to_string()));
        }
        if self.dtype() != DType::F32 || other.dtype() != DType::F32 {
            return Err(AnvilError::InvalidInput("SIMD ops only support F32 for now".to_string()));
        }

        let mut result = Self::new(self.shape(), DType::F32, self.device())?;
        let a_data = self.as_slice::<f32>();
        let b_data = other.as_slice::<f32>();
        let result_data = result.as_slice_mut::<f32>();

        // SIMD-accelerated addition
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                Self::add_simd_avx2(a_data, b_data, result_data);
            } else if is_x86_feature_detected!("sse4.1") {
                Self::add_simd_sse(a_data, b_data, result_data);
            } else {
                Self::add_scalar(a_data, b_data, result_data);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::add_simd_neon(a_data, b_data, result_data);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::add_scalar(a_data, b_data, result_data);
        }

        Ok(result)
    }

    /// SIMD-accelerated element-wise multiplication
    pub fn mul_simd(&self, other: &Self) -> AnvilResult<Self> {
        if self.shape() != other.shape() {
            return Err(AnvilError::InvalidInput("Shape mismatch".to_string()));
        }
        if self.dtype() != DType::F32 || other.dtype() != DType::F32 {
            return Err(AnvilError::InvalidInput("SIMD ops only support F32 for now".to_string()));
        }

        let mut result = Self::new(self.shape(), DType::F32, self.device())?;
        let a_data = self.as_slice::<f32>();
        let b_data = other.as_slice::<f32>();
        let result_data = result.as_slice_mut::<f32>();

        // SIMD-accelerated multiplication
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                Self::mul_simd_avx2(a_data, b_data, result_data);
            } else if is_x86_feature_detected!("sse4.1") {
                Self::mul_simd_sse(a_data, b_data, result_data);
            } else {
                Self::mul_scalar(a_data, b_data, result_data);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::mul_simd_neon(a_data, b_data, result_data);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::mul_scalar(a_data, b_data, result_data);
        }

        Ok(result)
    }

    /// SIMD-accelerated ReLU activation
    pub fn relu_simd(&self) -> AnvilResult<Self> {
        if self.dtype() != DType::F32 {
            return Err(AnvilError::InvalidInput("SIMD ops only support F32 for now".to_string()));
        }

        let mut result = Self::new(self.shape(), DType::F32, self.device())?;
        let input_data = self.as_slice::<f32>();
        let result_data = result.as_slice_mut::<f32>();

        // SIMD-accelerated ReLU
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                Self::relu_simd_avx2(input_data, result_data);
            } else if is_x86_feature_detected!("sse4.1") {
                Self::relu_simd_sse(input_data, result_data);
            } else {
                Self::relu_scalar(input_data, result_data);
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::relu_simd_neon(input_data, result_data);
        }

        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        {
            Self::relu_scalar(input_data, result_data);
        }

        Ok(result)
    }

    #[cfg(target_arch = "x86_64")]
    fn add_simd_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.chunks_exact(8);
        let b_chunks = b.chunks_exact(8);
        let result_chunks = result.chunks_exact_mut(8);

        for ((a_chunk, b_chunk), result_chunk) in chunks.zip(b_chunks).zip(result_chunks) {
            unsafe {
                let a_vec = _mm256_loadu_ps(a_chunk.as_ptr());
                let b_vec = _mm256_loadu_ps(b_chunk.as_ptr());
                let sum = _mm256_add_ps(a_vec, b_vec);
                _mm256_storeu_ps(result_chunk.as_mut_ptr(), sum);
            }
        }

        // Handle remainder
        let remainder_start = chunks.len() * 8;
        for i in remainder_start..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn add_simd_sse(a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.chunks_exact(4);
        let b_chunks = b.chunks_exact(4);
        let result_chunks = result.chunks_exact_mut(4);

        for ((a_chunk, b_chunk), result_chunk) in chunks.zip(b_chunks).zip(result_chunks) {
            unsafe {
                let a_vec = _mm_loadu_ps(a_chunk.as_ptr());
                let b_vec = _mm_loadu_ps(b_chunk.as_ptr());
                let sum = _mm_add_ps(a_vec, b_vec);
                _mm_storeu_ps(result_chunk.as_mut_ptr(), sum);
            }
        }

        // Handle remainder
        let remainder_start = chunks.len() * 4;
        for i in remainder_start..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn add_simd_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.chunks_exact(4);
        let b_chunks = b.chunks_exact(4);
        let result_chunks = result.chunks_exact_mut(4);

        for ((a_chunk, b_chunk), result_chunk) in chunks.clone().zip(b_chunks).zip(result_chunks) {
            unsafe {
                let a_vec = vld1q_f32(a_chunk.as_ptr());
                let b_vec = vld1q_f32(b_chunk.as_ptr());
                let sum = vaddq_f32(a_vec, b_vec);
                vst1q_f32(result_chunk.as_mut_ptr(), sum);
            }
        }

        // Handle remainder
        let remainder_start = chunks.len() * 4;
        for i in remainder_start..a.len() {
            result[i] = a[i] + b[i];
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn mul_simd_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.chunks_exact(8);
        let b_chunks = b.chunks_exact(8);
        let result_chunks = result.chunks_exact_mut(8);

        for ((a_chunk, b_chunk), result_chunk) in chunks.clone().zip(b_chunks).zip(result_chunks) {
            unsafe {
                let a_vec = _mm256_loadu_ps(a_chunk.as_ptr());
                let b_vec = _mm256_loadu_ps(b_chunk.as_ptr());
                let product = _mm256_mul_ps(a_vec, b_vec);
                _mm256_storeu_ps(result_chunk.as_mut_ptr(), product);
            }
        }

        // Handle remainder
        let remainder_start = chunks.len() * 8;
        for i in remainder_start..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn mul_simd_sse(a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.chunks_exact(4);
        let b_chunks = b.chunks_exact(4);
        let result_chunks = result.chunks_exact_mut(4);

        for ((a_chunk, b_chunk), result_chunk) in chunks.clone().zip(b_chunks).zip(result_chunks) {
            unsafe {
                let a_vec = _mm_loadu_ps(a_chunk.as_ptr());
                let b_vec = _mm_loadu_ps(b_chunk.as_ptr());
                let product = _mm_mul_ps(a_vec, b_vec);
                _mm_storeu_ps(result_chunk.as_mut_ptr(), product);
            }
        }

        // Handle remainder
        let remainder_start = chunks.len() * 4;
        for i in remainder_start..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn mul_simd_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
        let chunks = a.chunks_exact(4);
        let b_chunks = b.chunks_exact(4);
        let result_chunks = result.chunks_exact_mut(4);

        for ((a_chunk, b_chunk), result_chunk) in chunks.clone().zip(b_chunks).zip(result_chunks) {
            unsafe {
                let a_vec = vld1q_f32(a_chunk.as_ptr());
                let b_vec = vld1q_f32(b_chunk.as_ptr());
                let product = vmulq_f32(a_vec, b_vec);
                vst1q_f32(result_chunk.as_mut_ptr(), product);
            }
        }

        // Handle remainder
        let remainder_start = chunks.len() * 4;
        for i in remainder_start..a.len() {
            result[i] = a[i] * b[i];
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn relu_simd_avx2(input: &[f32], result: &mut [f32]) {
        let chunks = input.chunks_exact(8);
        let result_chunks = result.chunks_exact_mut(8);

        for (input_chunk, result_chunk) in chunks.clone().zip(result_chunks) {
            unsafe {
                let input_vec = _mm256_loadu_ps(input_chunk.as_ptr());
                let relu_vec = _mm256_max_ps(input_vec, _mm256_setzero_ps());
                _mm256_storeu_ps(result_chunk.as_mut_ptr(), relu_vec);
            }
        }

        // Handle remainder
        let remainder_start = chunks.len() * 8;
        for i in remainder_start..input.len() {
            result[i] = input[i].max(0.0);
        }
    }

    #[cfg(target_arch = "x86_64")]
    fn relu_simd_sse(input: &[f32], result: &mut [f32]) {
        let chunks = input.chunks_exact(4);
        let result_chunks = result.chunks_exact_mut(4);

        for (input_chunk, result_chunk) in chunks.clone().zip(result_chunks) {
            unsafe {
                let input_vec = _mm_loadu_ps(input_chunk.as_ptr());
                let relu_vec = _mm_max_ps(input_vec, _mm_setzero_ps());
                _mm_storeu_ps(result_chunk.as_mut_ptr(), relu_vec);
            }
        }

        // Handle remainder
        let remainder_start = chunks.len() * 4;
        for i in remainder_start..input.len() {
            result[i] = input[i].max(0.0);
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn relu_simd_neon(input: &[f32], result: &mut [f32]) {
        let chunks = input.chunks_exact(4);
        let result_chunks = result.chunks_exact_mut(4);

        for (input_chunk, result_chunk) in chunks.clone().zip(result_chunks) {
            unsafe {
                let input_vec = vld1q_f32(input_chunk.as_ptr());
                let zero_vec = vdupq_n_f32(0.0);
                let relu_vec = vmaxq_f32(input_vec, zero_vec);
                vst1q_f32(result_chunk.as_mut_ptr(), relu_vec);
            }
        }

        // Handle remainder
        let remainder_start = chunks.len() * 4;
        for i in remainder_start..input.len() {
            result[i] = input[i].max(0.0);
        }
    }
    
    /// SIMD-accelerated matrix multiplication for 2D tensors
    pub fn matmul_simd(&self, other: &Self) -> AnvilResult<Self> 
    where
        T: Copy + std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Default,
    {
        if DIMS != 2 {
            return Err(AnvilError::InvalidInput("Matrix multiplication only supports 2D tensors".to_string()));
        }
        
        let self_shape = self.shape();
        let other_shape = other.shape();
        
        if self_shape.dims[1] != other_shape.dims[0] {
            return Err(AnvilError::InvalidInput("Matrix dimensions incompatible for multiplication".to_string()));
        }
        
        // For now, delegate to linear ops for actual SIMD matmul
        // This would be implemented with specialized SIMD GEMM kernels in production
        Err(AnvilError::ComputationError("SIMD matmul not yet implemented".to_string()))
    }
    
    /// SIMD-accelerated reduction operations
    pub fn sum_simd(&self) -> AnvilResult<T> 
    where
        T: Copy + std::ops::Add<Output = T> + Default + PartialEq,
    {
        if self.dtype() != DType::F32 {
            return Err(AnvilError::InvalidInput("SIMD sum only supports F32 for now".to_string()));
        }
        
        let data = self.as_slice::<f32>();
        
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return Ok(unsafe { std::mem::transmute(Self::sum_simd_avx2(data)) });
            } else if is_x86_feature_detected!("sse4.1") {
                return Ok(unsafe { std::mem::transmute(Self::sum_simd_sse(data)) });
            }
        }
        
        #[cfg(target_arch = "aarch64")]
        {
            return Ok(unsafe { std::mem::transmute(Self::sum_simd_neon(data)) });
        }
        
        // Fallback to scalar implementation
        Ok(unsafe { std::mem::transmute(data.iter().fold(0.0f32, |acc, &x| acc + x)) })
    }
    
    /// Advanced SIMD sum reduction with AVX2
    #[cfg(target_arch = "x86_64")]
    unsafe fn sum_simd_avx2(data: &[f32]) -> f32 {
        let mut sum_vec = _mm256_setzero_ps();
        let chunks = data.chunks_exact(8);
        
        for chunk in chunks {
            let vec = _mm256_loadu_ps(chunk.as_ptr());
            sum_vec = _mm256_add_ps(sum_vec, vec);
        }
        
        // Horizontal sum of the AVX2 register
        let sum_high = _mm256_extractf128_ps(sum_vec, 1);
        let sum_low = _mm256_castps256_ps128(sum_vec);
        let sum_128 = _mm_add_ps(sum_high, sum_low);
        let sum_64 = _mm_add_ps(sum_128, _mm_movehl_ps(sum_128, sum_128));
        let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));
        
        let mut result = _mm_cvtss_f32(sum_32);
        
        // Add remainder
        let remainder_start = chunks.len() * 8;
        for &val in &data[remainder_start..] {
            result += val;
        }
        
        result
    }
    
    /// Advanced SIMD sum reduction with SSE
    #[cfg(target_arch = "x86_64")]
    unsafe fn sum_simd_sse(data: &[f32]) -> f32 {
        let mut sum_vec = _mm_setzero_ps();
        let chunks = data.chunks_exact(4);
        
        for chunk in chunks {
            let vec = _mm_loadu_ps(chunk.as_ptr());
            sum_vec = _mm_add_ps(sum_vec, vec);
        }
        
        // Horizontal sum
        let sum_64 = _mm_add_ps(sum_vec, _mm_movehl_ps(sum_vec, sum_vec));
        let sum_32 = _mm_add_ss(sum_64, _mm_shuffle_ps(sum_64, sum_64, 1));
        
        let mut result = _mm_cvtss_f32(sum_32);
        
        // Add remainder
        let remainder_start = chunks.len() * 4;
        for &val in &data[remainder_start..] {
            result += val;
        }
        
        result
    }
    
    /// Advanced SIMD sum reduction with NEON
    #[cfg(target_arch = "aarch64")]
    unsafe fn sum_simd_neon(data: &[f32]) -> f32 {
        let mut sum_vec = vdupq_n_f32(0.0);
        let chunks = data.chunks_exact(4);
        
        for chunk in chunks {
            let vec = vld1q_f32(chunk.as_ptr());
            sum_vec = vaddq_f32(sum_vec, vec);
        }
        
        // Horizontal sum
        let sum_pair = vpadd_f32(vget_low_f32(sum_vec), vget_high_f32(sum_vec));
        let sum_single = vpadd_f32(sum_pair, sum_pair);
        let mut result = vget_lane_f32(sum_single, 0);
        
        // Add remainder
        let remainder_start = chunks.len() * 4;
        for &val in &data[remainder_start..] {
            result += val;
        }
        
        result
    }

    fn add_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
        result.par_iter_mut()
            .zip(a.par_iter().zip(b.par_iter()))
            .for_each(|(r, (&a_val, &b_val))| {
                *r = a_val + b_val;
            });
    }

    fn mul_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
        result.par_iter_mut()
            .zip(a.par_iter().zip(b.par_iter()))
            .for_each(|(r, (&a_val, &b_val))| {
                *r = a_val * b_val;
            });
    }

    fn relu_scalar(input: &[f32], result: &mut [f32]) {
        result.par_iter_mut()
            .zip(input.par_iter())
            .for_each(|(r, &input_val)| {
                *r = input_val.max(0.0);
            });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_addition() {
        let shape = Shape::new([4, 4]);
        let data1 = vec![1.0; 16];
        let data2 = vec![2.0; 16];
        
        let tensor1 = AdvancedTensor::from_vec(data1, shape).unwrap();
        let tensor2 = AdvancedTensor::from_vec(data2, shape).unwrap();
        
        let result = tensor1.add_simd(&tensor2).unwrap();
        let result_data = result.as_slice::<f32>();
        
        for &val in result_data {
            assert_eq!(val, 3.0);
        }
    }

    #[test]
    fn test_simd_multiplication() {
        let shape = Shape::new([4, 4]);
        let data1 = vec![2.0; 16];
        let data2 = vec![3.0; 16];
        
        let tensor1 = AdvancedTensor::from_vec(data1, shape).unwrap();
        let tensor2 = AdvancedTensor::from_vec(data2, shape).unwrap();
        
        let result = tensor1.mul_simd(&tensor2).unwrap();
        let result_data = result.as_slice::<f32>();
        
        for &val in result_data {
            assert_eq!(val, 6.0);
        }
    }

    #[test]
    fn test_simd_relu() {
        let shape = Shape::new([4, 4]);
        let data = vec![-1.0, 0.0, 1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0, 11.0, -12.0, 13.0, -14.0];
        
        let tensor = AdvancedTensor::from_vec(data, shape).unwrap();
        let result = tensor.relu_simd().unwrap();
        let result_data = result.as_slice::<f32>();
        
        let expected = vec![0.0, 0.0, 1.0, 0.0, 3.0, 0.0, 5.0, 0.0, 7.0, 0.0, 9.0, 0.0, 11.0, 0.0, 13.0, 0.0];
        
        for (actual, expected) in result_data.iter().zip(expected.iter()) {
            assert_eq!(actual, expected);
        }
    }
} 