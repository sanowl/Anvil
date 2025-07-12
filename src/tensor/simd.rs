//! SIMD-accelerated tensor operations

use rayon::prelude::*;
use crate::error::{AnvilError, AnvilResult};
use super::core::{AdvancedTensor, Shape, DType};

// SIMD support
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

impl<const DIMS: usize> AdvancedTensor<DIMS> {
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