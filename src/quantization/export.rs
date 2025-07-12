//! Quantization export functionality

use crate::{
    error::{AnvilError, AnvilResult},
    tensor::core::{AdvancedTensor, Shape},
    quantization::core::{QuantizedTensor, QuantizationParams},
};

/// Export quantization parameters
pub fn export_quantization_params(tensor: &AdvancedTensor<f32, 2>, params: &QuantizationParams) -> AnvilResult<Vec<u8>> {
    let mut buffer = Vec::new();
    
    // Write quantization parameters
    buffer.extend_from_slice(&params.scale.to_le_bytes());
    buffer.extend_from_slice(&params.zero_point.to_le_bytes());
    buffer.extend_from_slice(&format!("{:?}", params.dtype).as_bytes());
    
    // Write tensor metadata
    let shape = tensor.shape();
    buffer.extend_from_slice(&(shape.dims.len() as u32).to_le_bytes());
    for dim in shape.dims {
        buffer.extend_from_slice(&dim.to_le_bytes());
    }
    
    // Write quantized data
    let quantized = QuantizedTensor::from_tensor(tensor, params.clone())?;
    let quantized_data: Vec<u8> = quantized.data.iter().map(|&x| x as u8).collect();
    buffer.extend_from_slice(&quantized_data);
    
    Ok(buffer)
}

/// Import quantization parameters
pub fn import_quantization_params(data: &[u8]) -> AnvilResult<(QuantizedTensor, QuantizationParams)> {
    let mut offset = 0;
    
    // Read quantization parameters
    let scale = f32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
    offset += 4;
    
    let zero_point = i32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]);
    offset += 4;
    
    let dtype_str = String::from_utf8_lossy(&data[offset..offset+8]).to_string();
    offset += 8;
    
    let params = QuantizationParams {
        scale: 1.0,
        zero_point: 0,
        scheme: crate::quantization::core::QuantizationScheme::Symmetric,
        bits: 8,
        min_val: -1.0,
        max_val: 1.0,
        dtype: crate::tensor::core::DType::F32,
    };
    
    // Read tensor metadata
    let num_dims = u32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]) as usize;
    offset += 4;
    
    let mut dims = Vec::new();
    for _ in 0..num_dims {
        let dim = u32::from_le_bytes([data[offset], data[offset+1], data[offset+2], data[offset+3]]) as usize;
        offset += 4;
        dims.push(dim);
    }
    
    // Read quantized data
    let quantized_data = data[offset..].to_vec();
    
    let quantized_tensor = QuantizedTensor {
        data: quantized_data.into_iter().map(|x| x as i8).collect(),
        shape: Shape::new(dims.try_into().unwrap_or([0, 0])),
        params: params.clone(),
        device: crate::tensor::devices::Device::Cpu,
    };
    
    Ok((quantized_tensor, params))
}

/// Export to ONNX format
pub fn export_to_onnx(tensor: &AdvancedTensor<f32, 2>, params: &QuantizationParams) -> AnvilResult<Vec<u8>> {
    // Simplified ONNX export
    let mut onnx_data = Vec::new();
    
    // ONNX header
    onnx_data.extend_from_slice(b"ONNX");
    onnx_data.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // Version
    
    // Node metadata
    onnx_data.extend_from_slice(b"QuantizeLinear");
    
    // Export quantization parameters
    let quantized_data = export_quantization_params(tensor, params)?;
    onnx_data.extend_from_slice(&quantized_data);
    
    Ok(onnx_data)
}

/// Export to TensorRT format
pub fn export_to_tensorrt(tensor: &AdvancedTensor<f32, 2>, params: &QuantizationParams) -> AnvilResult<Vec<u8>> {
    // Simplified TensorRT export
    let mut trt_data = Vec::new();
    
    // TensorRT header
    trt_data.extend_from_slice(b"TRT");
    trt_data.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // Version
    
    // Export quantization parameters
    let quantized_data = export_quantization_params(tensor, params)?;
    trt_data.extend_from_slice(&quantized_data);
    
    Ok(trt_data)
}

/// Export to PyTorch format
pub fn export_to_pytorch(tensor: &AdvancedTensor<f32, 2>, params: &QuantizationParams) -> AnvilResult<Vec<u8>> {
    // Simplified PyTorch export
    let mut pt_data = Vec::new();
    
    // PyTorch header
    pt_data.extend_from_slice(b"PT");
    pt_data.extend_from_slice(&[0x01, 0x00, 0x00, 0x00]); // Version
    
    // Export quantization parameters
    let quantized_data = export_quantization_params(tensor, params)?;
    pt_data.extend_from_slice(&quantized_data);
    
    Ok(pt_data)
}

/// Compare original and quantized tensors
pub fn compare_quantization_quality(original: &AdvancedTensor<f32, 2>, quantized: &AdvancedTensor<f32, 2>) -> AnvilResult<QuantizationQuality> {
    let original_data = original.as_slice::<f32>();
    let quantized_data = quantized.as_slice::<f32>();
    
    let mut mse = 0.0f32;
    let mut mae = 0.0f32;
    let mut max_error = 0.0f32;
    
    for (i, (&orig, &quant)) in original_data.iter().zip(quantized_data.iter()).enumerate() {
        let diff = orig - quant;
        mse += diff * diff;
        mae += diff.abs();
        max_error = max_error.max(diff.abs());
    }
    
    let num_elements = original_data.len() as f32;
    mse /= num_elements;
    mae /= num_elements;
    
    // Calculate compression ratio
    let original_size = original_data.len() * 4; // 4 bytes per f32
    let quantized_size = original_data.len(); // 1 byte per quantized value
    let compression_ratio = original_size as f32 / quantized_size as f32;
    
    Ok(QuantizationQuality {
        mse,
        mae,
        max_error,
        compression_ratio,
        psnr: 20.0 * (1.0 / mse.sqrt()).log10(),
    })
}

/// Quantization quality metrics
#[derive(Debug, Clone)]
pub struct QuantizationQuality {
    pub mse: f32,
    pub mae: f32,
    pub max_error: f32,
    pub compression_ratio: f32,
    pub psnr: f32,
}

impl QuantizationQuality {
    pub fn is_acceptable(&self, threshold: f32) -> bool {
        self.mse < threshold
    }
    
    pub fn print_report(&self) {
        println!("Quantization Quality Report:");
        println!("  MSE: {:.6}", self.mse);
        println!("  MAE: {:.6}", self.mae);
        println!("  Max Error: {:.6}", self.max_error);
        println!("  Compression Ratio: {:.2}x", self.compression_ratio);
        println!("  PSNR: {:.2} dB", self.psnr);
    }
} 