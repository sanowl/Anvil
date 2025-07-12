//! Python bindings using PyO3

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use crate::tensor::core::{AdvancedTensor, Shape, DType};
use crate::error::AnvilResult;

/// Python wrapper for Anvil tensors
#[pyclass]
pub struct PyTensor {
    tensor: AdvancedTensor<f32, 2>,
}

#[pymethods]
impl PyTensor {
    #[new]
    fn new(data: Vec<f32>, shape: Vec<usize>) -> PyResult<Self> {
        if shape.len() != 2 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Only 2D tensors supported for Python"
            ));
        }
        
        let tensor_shape = Shape::new([shape[0], shape[1]]);
        let tensor = AdvancedTensor::<f32, 2>::from_vec(data, tensor_shape)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        
        Ok(Self { tensor })
    }
    
    fn shape(&self) -> PyResult<Vec<usize>> {
        Ok(self.tensor.shape().dims.to_vec())
    }
    
    fn data(&self) -> PyResult<Vec<f32>> {
        Ok(self.tensor.as_slice::<f32>().to_vec())
    }
    
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("PyTensor(shape={:?}, data={:?})", self.shape()?, self.data()?))
    }
    
    fn add(&self, other: &PyTensor) -> PyResult<PyTensor> {
        // Simple element-wise addition implementation for Python demo
        let a_data = self.tensor.as_slice::<f32>();
        let b_data = other.tensor.as_slice::<f32>();
        let a_shape = self.tensor.shape().dims;
        let b_shape = other.tensor.shape().dims;
        
        if a_shape != b_shape {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Tensor shapes must match for addition"
            ));
        }
        
        let result_data: Vec<f32> = a_data.iter().zip(b_data.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        PyTensor::new(result_data, a_shape.to_vec())
    }
    
    fn matmul(&self, other: &PyTensor) -> PyResult<PyTensor> {
        // Simple matrix multiplication implementation
        let a_data = self.tensor.as_slice::<f32>();
        let b_data = other.tensor.as_slice::<f32>();
        let a_shape = self.tensor.shape().dims;
        let b_shape = other.tensor.shape().dims;
        
        if a_shape[1] != b_shape[0] {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Matrix dimensions don't match for multiplication"
            ));
        }
        
        let m = a_shape[0];
        let k = a_shape[1];
        let n = b_shape[1];
        
        let mut result_data = vec![0.0; m * n];
        
        for i in 0..m {
            for j in 0..n {
                for l in 0..k {
                    result_data[i * n + j] += a_data[i * k + l] * b_data[l * n + j];
                }
            }
        }
        
        PyTensor::new(result_data, vec![m, n])
    }
    
    fn relu(&self) -> PyResult<PyTensor> {
        let data = self.tensor.as_slice::<f32>();
        let result_data: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
        let shape = self.tensor.shape().dims.to_vec();
        
        PyTensor::new(result_data, shape)
    }
    
    fn sigmoid(&self) -> PyResult<PyTensor> {
        let data = self.tensor.as_slice::<f32>();
        let result_data: Vec<f32> = data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let shape = self.tensor.shape().dims.to_vec();
        
        PyTensor::new(result_data, shape)
    }
}

/// Simple neural network for Python demo
#[pyclass]
pub struct PyNeuralNetwork {
    weights1: PyTensor,
    bias1: PyTensor,
    weights2: PyTensor, 
    bias2: PyTensor,
}

#[pymethods]
impl PyNeuralNetwork {
    #[new]
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> PyResult<Self> {
        // Initialize weights with small random values
        let w1_data: Vec<f32> = (0..input_size * hidden_size)
            .map(|_| fastrand::f32() * 0.1 - 0.05)
            .collect();
        let b1_data: Vec<f32> = vec![0.0; hidden_size];
        
        let w2_data: Vec<f32> = (0..hidden_size * output_size)
            .map(|_| fastrand::f32() * 0.1 - 0.05)
            .collect();
        let b2_data: Vec<f32> = vec![0.0; output_size];
        
        Ok(Self {
            weights1: PyTensor::new(w1_data, vec![input_size, hidden_size])?,
            bias1: PyTensor::new(b1_data, vec![1, hidden_size])?,
            weights2: PyTensor::new(w2_data, vec![hidden_size, output_size])?,
            bias2: PyTensor::new(b2_data, vec![1, output_size])?,
        })
    }
    
    fn forward(&self, input: &PyTensor) -> PyResult<PyTensor> {
        // Layer 1: input @ weights1 + bias1
        let hidden = input.matmul(&self.weights1)?;
        let hidden = hidden.add(&self.bias1)?;
        let hidden = hidden.relu()?;
        
        // Layer 2: hidden @ weights2 + bias2  
        let output = hidden.matmul(&self.weights2)?;
        let output = output.add(&self.bias2)?;
        let output = output.sigmoid()?;
        
        Ok(output)
    }
    
    fn __str__(&self) -> PyResult<String> {
        Ok("PyNeuralNetwork(2-layer feedforward network)".to_string())
    }
}

/// Python module for Anvil ML framework
#[pymodule]
fn anvil(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
    m.add_class::<PyNeuralNetwork>()?;
    m.add_function(wrap_pyfunction!(create_tensor, m)?)?;
    m.add_function(wrap_pyfunction!(test_framework, m)?)?;
    Ok(())
}

#[pyfunction]
fn create_tensor(data: Vec<f32>, shape: Vec<usize>) -> PyResult<PyTensor> {
    PyTensor::new(data, shape)
}

#[pyfunction]
fn test_framework() -> PyResult<String> {
    Ok("Anvil ML Framework is working! 🚀".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_py_tensor() {
        let tensor = PyTensor::new(vec![1.0f32, 2.0, 3.0, 4.0], vec![2, 2]);
        assert!(tensor.is_ok());
    }
    
    #[test]
    fn test_py_neural_network() {
        let model = PyNeuralNetwork::new(4, 8, 2);
        assert!(model.is_ok());
    }
    
    #[test]
    fn test_py_tensor_operations() {
        let t1 = PyTensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let t2 = PyTensor::new(vec![0.5, 1.0, 1.5, 2.0], vec![2, 2]).unwrap();
        
        let result = t1.add(&t2);
        assert!(result.is_ok());
        
        let matmul_result = t1.matmul(&t2);
        assert!(matmul_result.is_ok());
    }
} 