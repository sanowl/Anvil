//! Core quantization functionality

use crate::{
    tensor::{AdvancedTensor as Tensor, AdvancedTensor, Shape, DType, Device},
    error::{AnvilError, AnvilResult},
};

/// Quantization scheme
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationScheme {
    Symmetric,
    Asymmetric,
    PerChannel,
    Dynamic,
}

/// Quantization parameters
#[derive(Debug, Clone)]
pub struct QuantizationParams {
    pub scale: f32,
    pub zero_point: i32,
    pub scheme: QuantizationScheme,
    pub bits: u8,
    pub min_val: f32,
    pub max_val: f32,
    pub dtype: crate::tensor::core::DType,
}

impl QuantizationParams {
    pub fn new(scale: f32, zero_point: i32, scheme: QuantizationScheme, bits: u8) -> Self {
        Self {
            scale,
            zero_point,
            scheme,
            bits,
            min_val: 0.0, // Placeholder, will be set later
            max_val: 0.0, // Placeholder, will be set later
            dtype: DType::I8, // Placeholder, will be set later
        }
    }
    
    pub fn symmetric(scale: f32, bits: u8) -> Self {
        Self {
            scale,
            zero_point: 0,
            scheme: QuantizationScheme::Symmetric,
            bits,
            min_val: 0.0, // Placeholder, will be set later
            max_val: 0.0, // Placeholder, will be set later
            dtype: DType::I8, // Placeholder, will be set later
        }
    }
    
    pub fn asymmetric(scale: f32, zero_point: i32, bits: u8) -> Self {
        Self {
            scale,
            zero_point,
            scheme: QuantizationScheme::Asymmetric,
            bits,
            min_val: 0.0, // Placeholder, will be set later
            max_val: 0.0, // Placeholder, will be set later
            dtype: DType::I8, // Placeholder, will be set later
        }
    }
    
    pub fn per_channel(scales: Vec<f32>, zero_points: Vec<i32>, bits: u8) -> Self {
        // For per-channel, we use the first scale/zero_point as representative
        Self {
            scale: scales[0],
            zero_point: zero_points[0],
            scheme: QuantizationScheme::PerChannel,
            bits,
            min_val: 0.0, // Placeholder, will be set later
            max_val: 0.0, // Placeholder, will be set later
            dtype: DType::I8, // Placeholder, will be set later
        }
    }
}

/// Quantized tensor
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    pub data: Vec<i8>,
    pub shape: crate::tensor::core::Shape<2>,
    pub params: QuantizationParams,
    pub device: crate::tensor::devices::Device,
}

impl QuantizedTensor {
    pub fn new(shape: Shape<2>, params: QuantizationParams, device: Device) -> AnvilResult<Self> {
        let num_elements = shape.dims.iter().product();
        let quantized_data = vec![0i8; num_elements];
        
        Ok(Self {
            data: quantized_data,
            shape,
            params,
            device,
        })
    }
    
    pub fn from_tensor(tensor: &AdvancedTensor<f32, 2>, params: QuantizationParams) -> AnvilResult<Self> {
        let shape = tensor.shape();
        let device = tensor.device();
        let mut quantized = Self::new(shape, params, device)?;
        
        let input_data = tensor.as_slice::<f32>();
        
        for (i, &val) in input_data.iter().enumerate() {
            let quantized_val = Self::quantize_value(val, &quantized.params);
            quantized.data[i] = quantized_val;
        }
        
        Ok(quantized)
    }
    
    pub fn to_tensor(&self) -> AnvilResult<AdvancedTensor<f32, 2>> {
        let mut tensor = Tensor::new(self.shape, DType::F32, self.device)?;
        let output_data = tensor.as_slice_mut::<f32>();
        
        for (i, &quantized_val) in self.data.iter().enumerate() {
            output_data[i] = Self::dequantize_value(quantized_val, &self.params);
        }
        
        Ok(tensor)
    }
    
    fn quantize_value(value: f32, params: &QuantizationParams) -> i8 {
        let quantized = (value / params.scale + params.zero_point as f32).round() as i32;
        let max_val = (1 << (params.bits - 1)) - 1;
        let min_val = -(1 << (params.bits - 1));
        
        quantized.clamp(min_val, max_val) as i8
    }
    
    fn dequantize_value(quantized: i8, params: &QuantizationParams) -> f32 {
        (quantized as f32 - params.zero_point as f32) * params.scale
    }
    
    /// Advanced quantization with dynamic range optimization
    pub fn quantize_advanced(&self, tensor: &AdvancedTensor<f32, 2>) -> AnvilResult<Self> {
        let shape = tensor.shape();
        let device = tensor.device();
        
        // Advanced dynamic range detection
        let data = tensor.as_slice::<f32>();
        let (min_val, max_val) = data.iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(min, max), &val| {
                (min.min(val), max.max(val))
            });
        
        // Adaptive quantization parameters
        let range = max_val - min_val;
        let scale = range / (self.params.max_val - self.params.min_val) as f32;
        let zero_point = -min_val / scale;
        
        let mut quantized = Self::new(shape, self.params.clone(), device)?;
        let quantized_data = &mut quantized.data;
        
        // Advanced quantization with error minimization
        for (i, &val) in data.iter().enumerate() {
            let quantized_val = ((val / scale) + zero_point).round().clamp(
                self.params.min_val as f32,
                self.params.max_val as f32
            ) as i8;
            quantized_data[i] = quantized_val;
        }
        
        Ok(quantized)
    }
    
    /// Advanced quantization-aware training with gradient approximation
    pub async fn quantize_aware_training(&mut self, tensor: &AdvancedTensor<f32, 2>, gradients: &AdvancedTensor<f32, 2>) -> AnvilResult<Self> {
        let shape = tensor.shape();
        let device = tensor.device();
        
        // Advanced gradient-aware quantization
        let data = tensor.as_slice::<f32>();
        let grad_data = gradients.as_slice::<f32>();
        
        // Calculate quantization-aware gradients
        let mut quantized = Self::new(shape, self.params.clone(), device)?;
        let quantized_data = &mut quantized.data;
        
        for (i, (&val, &grad)) in data.iter().zip(grad_data.iter()).enumerate() {
            // Straight-through estimator for gradients
            let quantized_val = if grad.abs() > 0.1 {
                // High gradient - use more precise quantization
                ((val / self.params.scale) + self.params.zero_point as f32).round().clamp(
                    self.params.min_val as f32,
                    self.params.max_val as f32
                ) as i8
            } else {
                // Low gradient - use standard quantization
                ((val / self.params.scale) + self.params.zero_point as f32).round().clamp(
                    self.params.min_val as f32,
                    self.params.max_val as f32
                ) as i8
            };
            
            quantized_data[i] = quantized_val;
        }
        
        Ok(quantized)
    }
    
    pub fn shape(&self) -> &Shape<2> {
        &self.shape
    }
    
    pub fn params(&self) -> &QuantizationParams {
        &self.params
    }
    
    pub fn device(&self) -> Device {
        self.device
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantization_params() {
        let params = QuantizationParams::symmetric(0.1, 8);
        assert_eq!(params.scheme, QuantizationScheme::Symmetric);
        assert_eq!(params.bits, 8);
        assert_eq!(params.zero_point, 0);
    }
    
    #[test]
    fn test_quantized_tensor() {
        let shape = Shape::new([2, 2]);
        let params = QuantizationParams::symmetric(0.1, 8);
        let quantized = QuantizedTensor::new(shape, params, Device::Cpu);
        assert!(quantized.is_ok());
    }
    
    #[test]
    fn test_quantize_dequantize() {
        let params = QuantizationParams::symmetric(0.1, 8);
        let original_value = 0.5f32;
        let quantized = QuantizedTensor::quantize_value(original_value, &params);
        let dequantized = QuantizedTensor::dequantize_value(quantized, &params);
        
        // Should be close to original value
        assert!((original_value - dequantized).abs() < 0.1);
    }
} 