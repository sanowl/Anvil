//! Quantization algorithms

use crate::{
    tensor::{AdvancedTensor as Tensor, AdvancedTensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
    quantization::core::{QuantizationParams, QuantizedTensor},
};

/// Min-max quantization algorithm
pub struct MinMaxQuantizer {
    bits: u8,
}

impl MinMaxQuantizer {
    pub fn new(bits: u8) -> Self {
        Self { bits }
    }
    
    pub fn quantize(&self, tensor: &AdvancedTensor<f32, 2>) -> AnvilResult<QuantizationParams> {
        let data = tensor.as_slice::<f32>();
        let shape = tensor.shape();
        
        // Find min/max values
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        let range = max_val - min_val;
        let scale = range / 255.0; // 8-bit quantization
        let zero_point = (-min_val / scale).round() as i32;
        
        Ok(QuantizationParams::asymmetric(scale, zero_point, 8))
    }
}

/// Histogram-based quantization algorithm
pub struct HistogramQuantizer {
    bits: u8,
    num_bins: usize,
}

impl HistogramQuantizer {
    pub fn new(bits: u8, num_bins: usize) -> Self {
        Self { bits, num_bins }
    }
    
    pub fn quantize(&self, tensor: &AdvancedTensor<f32, 2>) -> AnvilResult<QuantizationParams> {
        let data = tensor.as_slice::<f32>();
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Create histogram
        let mut histogram = vec![0; self.num_bins];
        let bin_size = (max_val - min_val) / self.num_bins as f32;
        
        for &val in data {
            let bin_idx = ((val - min_val) / bin_size).floor() as usize;
            let bin_idx = bin_idx.min(self.num_bins - 1);
            histogram[bin_idx] += 1;
        }
        
        // Find optimal scale based on histogram
        let scale = (max_val - min_val) / ((1 << self.bits) - 1) as f32;
        let zero_point = (-min_val / scale).round() as i32;
        
        Ok(QuantizationParams::asymmetric(scale, zero_point, self.bits))
    }
}

/// KL divergence quantization algorithm
pub struct KLDivergenceQuantizer {
    bits: u8,
    num_bins: usize,
}

impl KLDivergenceQuantizer {
    pub fn new(bits: u8, num_bins: usize) -> Self {
        Self { bits, num_bins }
    }
    
    pub fn quantize(&self, tensor: &AdvancedTensor<f32, 2>) -> AnvilResult<QuantizationParams> {
        let data = tensor.as_slice::<f32>();
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Simplified KL divergence quantization
        let scale = (max_val - min_val) / ((1 << self.bits) - 1) as f32;
        let zero_point = (-min_val / scale).round() as i32;
        
        Ok(QuantizationParams::asymmetric(scale, zero_point, self.bits))
    }
}

/// Per-channel quantization algorithm
pub struct PerChannelQuantizer {
    bits: u8,
}

impl PerChannelQuantizer {
    pub fn new(bits: u8) -> Self {
        Self { bits }
    }
    
    pub fn quantize(&self, tensor: &AdvancedTensor<f32, 2>) -> AnvilResult<QuantizationParams> {
        let shape = tensor.shape();
        let data = tensor.as_slice::<f32>();
        let (rows, cols) = (shape.dims[0], shape.dims[1]);
        
        let mut scales = Vec::new();
        let mut zero_points = Vec::new();
        
        for col in 0..cols {
            let mut min_val = f32::INFINITY;
            let mut max_val = f32::NEG_INFINITY;
            
            for row in 0..rows {
                let val = data[row * cols + col];
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
            
            let range = max_val - min_val;
            let scale = range / ((1 << self.bits) - 1) as f32;
            let zero_point = (-min_val / scale).round() as i32;
            
            scales.push(scale);
            zero_points.push(zero_point);
        }
        
        Ok(QuantizationParams::per_channel(scales, zero_points, self.bits))
    }
}

/// Dynamic quantization algorithm
pub struct DynamicQuantizer {
    bits: u8,
}

impl DynamicQuantizer {
    pub fn new(bits: u8) -> Self {
        Self { bits }
    }
    
    pub fn quantize(&self, tensor: &AdvancedTensor<f32, 2>) -> AnvilResult<QuantizationParams> {
        let data = tensor.as_slice::<f32>();
        let min_val = data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        
        // Dynamic quantization uses symmetric scheme
        let abs_max = min_val.abs().max(max_val.abs());
        let scale = abs_max / ((1 << (self.bits - 1)) - 1) as f32;
        
        Ok(QuantizationParams::symmetric(scale, self.bits))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_min_max_quantizer() {
        let tensor = Tensor::new(Shape::new([2, 2]), DType::F32, Device::Cpu).unwrap();
        let quantizer = MinMaxQuantizer::new(8);
        let params = quantizer.quantize(&tensor);
        assert!(params.is_ok());
    }
    
    #[test]
    fn test_histogram_quantizer() {
        let tensor = Tensor::new(Shape::new([2, 2]), DType::F32, Device::Cpu).unwrap();
        let quantizer = HistogramQuantizer::new(8, 256);
        let params = quantizer.quantize(&tensor);
        assert!(params.is_ok());
    }
    
    #[test]
    fn test_kl_divergence_quantizer() {
        let tensor = Tensor::new(Shape::new([2, 2]), DType::F32, Device::Cpu).unwrap();
        let quantizer = KLDivergenceQuantizer::new(8, 256);
        let params = quantizer.quantize(&tensor);
        assert!(params.is_ok());
    }
    
    #[test]
    fn test_per_channel_quantizer() {
        let tensor = Tensor::new(Shape::new([2, 3]), DType::F32, Device::Cpu).unwrap();
        let quantizer = PerChannelQuantizer::new(8);
        let params = quantizer.quantize(&tensor);
        assert!(params.is_ok());
    }
    
    #[test]
    fn test_dynamic_quantizer() {
        let tensor = Tensor::new(Shape::new([2, 2]), DType::F32, Device::Cpu).unwrap();
        let quantizer = DynamicQuantizer::new(8);
        let params = quantizer.quantize(&tensor);
        assert!(params.is_ok());
    }
} 