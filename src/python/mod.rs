//! Python bindings using PyO3

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use crate::tensor::core::{AdvancedTensor, Shape, DType};
use crate::error::AnvilResult;

/// Python wrapper for Anvil tensors
#[pyclass]
pub struct PyTensor {
    tensor: AdvancedTensor<2>,
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
        let result = self.tensor.add(&other.tensor)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        
        let data = result.as_slice::<f32>().to_vec();
        let shape = result.shape().dims.to_vec();
        
        PyTensor::new(data, shape)
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
}

/// Python module for Anvil ML framework
#[pymodule]
fn anvil(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyTensor>()?;
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
    fn test_py_model() {
        let model = PyFeedForwardNet::new(784, 10);
        assert!(model.is_ok());
    }
} 