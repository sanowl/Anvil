//! Advanced GPU compute engine with multiple backend support

use std::sync::Arc;
use std::collections::HashMap;
use wgpu::{Device, Queue, Buffer, ComputePipeline, BindGroup, CommandEncoder};
use crate::{
    tensor::AdvancedTensor,
    error::{AnvilError, AnvilResult},
    gpu::unified::UnifiedGPUBackend,
};

/// GPU compute kernel types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelType {
    MatMulTiled,
    MatMulBatched,
    ElementwiseAdd,
    ElementwiseMul,
    ElementwiseRelu,
    ElementwiseGelu,
    ElementwiseSwish,
    ReduceSum,
    ReduceMax,
    Conv2D,
    DepthwiseConv2D,
    Softmax,
    MultiHeadAttention,
    LayerNorm,
}

/// GPU buffer with metadata
#[derive(Debug)]
pub struct GPUBuffer {
    buffer: Buffer,
    size: u64,
    usage: wgpu::BufferUsages,
    label: String,
}

impl GPUBuffer {
    pub fn new(device: &Device, size: u64, usage: wgpu::BufferUsages, label: &str) -> Self {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        });
        
        Self {
            buffer,
            size,
            usage,
            label: label.to_string(),
        }
    }
    
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }
    
    pub fn size(&self) -> u64 {
        self.size
    }
    
    pub fn write_data<T>(&self, queue: &Queue, data: &[T]) 
    where
        T: bytemuck::Pod,
    {
        queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(data));
    }
    
    pub async fn read_data<T>(&self, device: &Device) -> AnvilResult<Vec<T>>
    where
        T: bytemuck::Pod + Clone + Default,
    {
        // Create staging buffer for readback
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging_buffer"),
            size: self.size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // Copy GPU buffer to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("copy_encoder"),
        });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &staging_buffer, 0, self.size);
        let command_buffer = encoder.finish();
        
        // Submit and wait for completion
        let queue = device.as_ref(); // This is simplified - need proper queue access
        // queue.submit(std::iter::once(command_buffer));
        
        // Map and read data
        let buffer_slice = staging_buffer.slice(..);
        let (tx, rx) = futures_intrusive::channel::shared::oneshot_channel();
        
        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        
        // device.poll(wgpu::Maintain::Wait);
        rx.receive().await.unwrap().map_err(|e| AnvilError::ComputationError(format!("Buffer mapping failed: {:?}", e)))?;
        
        let data = buffer_slice.get_mapped_range();
        let typed_data: &[T] = bytemuck::cast_slice(&data);
        let result = typed_data.to_vec();
        
        drop(data);
        staging_buffer.unmap();
        
        Ok(result)
    }
}

/// Advanced GPU compute engine
pub struct ComputeEngine {
    device: Arc<Device>,
    queue: Arc<Queue>,
    pipelines: HashMap<KernelType, ComputePipeline>,
    shader_modules: HashMap<KernelType, wgpu::ShaderModule>,
    buffer_pool: Vec<GPUBuffer>,
}

impl ComputeEngine {
    pub async fn new() -> AnvilResult<Self> {
        // Initialize WGPU instance
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            dx12_shader_compiler: Default::default(),
            flags: wgpu::InstanceFlags::default(),
            gles_minor_version: wgpu::Gles3MinorVersion::Automatic,
        });
        
        // Request adapter
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| AnvilError::ComputationError("Failed to find adapter".to_string()))?;
        
        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Anvil Compute Device"),
                    features: wgpu::Features::empty(),
                    limits: wgpu::Limits::default(),
                },
                None,
            )
            .await
            .map_err(|e| AnvilError::ComputationError(format!("Failed to create device: {:?}", e)))?;
        
        let device = Arc::new(device);
        let queue = Arc::new(queue);
        
        let mut engine = Self {
            device: device.clone(),
            queue,
            pipelines: HashMap::new(),
            shader_modules: HashMap::new(),
            buffer_pool: Vec::new(),
        };
        
        // Load and compile all shaders
        engine.initialize_shaders()?;
        
        Ok(engine)
    }
    
    fn initialize_shaders(&mut self) -> AnvilResult<()> {
        // Load the advanced kernels shader
        let shader_source = include_str!("../kernels/advanced_kernels.wgsl");
        
        let shader_module = self.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("advanced_kernels"),
            source: wgpu::ShaderSource::Wgsl(shader_source.into()),
        });
        
        // Create compute pipelines for each kernel type
        let kernel_entry_points = [
            (KernelType::MatMulTiled, "matmul_tiled"),
            (KernelType::MatMulBatched, "matmul_batched"),
            (KernelType::ElementwiseAdd, "elementwise_add"),
            (KernelType::ElementwiseMul, "elementwise_mul"),
            (KernelType::ElementwiseRelu, "elementwise_relu"),
            (KernelType::ElementwiseGelu, "elementwise_gelu"),
            (KernelType::ElementwiseSwish, "elementwise_swish"),
            (KernelType::ReduceSum, "reduce_sum"),
            (KernelType::ReduceMax, "reduce_max"),
            (KernelType::Conv2D, "conv2d"),
            (KernelType::DepthwiseConv2D, "depthwise_conv2d"),
            (KernelType::Softmax, "softmax"),
            (KernelType::MultiHeadAttention, "multi_head_attention"),
            (KernelType::LayerNorm, "layer_norm"),
        ];
        
        for (kernel_type, entry_point) in kernel_entry_points {
            let bind_group_layout = self.create_bind_group_layout_for_kernel(kernel_type);
            
            let pipeline_layout = self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(&format!("{:?}_pipeline_layout", kernel_type)),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });
            
            let pipeline = self.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{:?}_pipeline", kernel_type)),
                layout: Some(&pipeline_layout),
                module: &shader_module,
                entry_point,
            });
            
            self.pipelines.insert(kernel_type, pipeline);
            self.shader_modules.insert(kernel_type, shader_module.clone());
        }
        
        Ok(())
    }
    
    fn create_bind_group_layout_for_kernel(&self, kernel_type: KernelType) -> wgpu::BindGroupLayout {
        match kernel_type {
            KernelType::MatMulTiled | KernelType::MatMulBatched => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("matmul_bind_group_layout"),
                    entries: &[
                        // Matrix A (input)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Matrix B (input)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Matrix C (output)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Dimensions uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            },
            KernelType::ElementwiseAdd | KernelType::ElementwiseMul | 
            KernelType::ElementwiseRelu | KernelType::ElementwiseGelu | 
            KernelType::ElementwiseSwish => {
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("elementwise_bind_group_layout"),
                    entries: &[
                        // Input A
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Input B (optional for unary operations)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Output
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Size uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                })
            },
            _ => {
                // Default layout for other kernels
                self.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("default_bind_group_layout"),
                    entries: &[],
                })
            }
        }
    }
    
    /// Execute matrix multiplication on GPU
    pub async fn matmul<const DIMS: usize>(
        &mut self,
        a: &AdvancedTensor<f32, DIMS>,
        b: &AdvancedTensor<f32, DIMS>,
        tiled: bool,
    ) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        if DIMS != 2 {
            return Err(AnvilError::InvalidInput("Matrix multiplication requires 2D tensors".to_string()));
        }
        
        let a_shape = a.shape();
        let b_shape = b.shape();
        
        if a_shape.dims[1] != b_shape.dims[0] {
            return Err(AnvilError::InvalidInput("Matrix dimension mismatch".to_string()));
        }
        
        let m = a_shape.dims[0] as u32;
        let k = a_shape.dims[1] as u32;
        let n = b_shape.dims[1] as u32;
        
        // Create GPU buffers
        let a_buffer = self.create_buffer_from_tensor(a, "matrix_a")?;
        let b_buffer = self.create_buffer_from_tensor(b, "matrix_b")?;
        
        let result_size = (m * n) as u64 * std::mem::size_of::<f32>() as u64;
        let c_buffer = GPUBuffer::new(
            &self.device,
            result_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "matrix_c",
        );
        
        // Create dimensions uniform buffer
        let dimensions = [m, k, n, 1u32]; // batch_size = 1 for basic matmul
        let dimensions_buffer = self.create_uniform_buffer(&dimensions, "dimensions")?;
        
        // Choose kernel type
        let kernel_type = if tiled {
            KernelType::MatMulTiled
        } else {
            KernelType::MatMulBatched
        };
        
        // Create bind group
        let bind_group_layout = self.create_bind_group_layout_for_kernel(kernel_type);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("matmul_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: c_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: dimensions_buffer.buffer().as_entire_binding(),
                },
            ],
        });
        
        // Dispatch compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("matmul_encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("matmul_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(self.pipelines.get(&kernel_type).unwrap());
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            // Calculate dispatch size
            let workgroup_size_x = 16;
            let workgroup_size_y = 16;
            let dispatch_x = (m + workgroup_size_x - 1) / workgroup_size_x;
            let dispatch_y = (n + workgroup_size_y - 1) / workgroup_size_y;
            
            compute_pass.dispatch_workgroups(dispatch_x, dispatch_y, 1);
        }
        
        let command_buffer = encoder.finish();
        self.queue.submit(std::iter::once(command_buffer));
        
        // Read back result
        let result_data: Vec<f32> = c_buffer.read_data(&self.device).await?;
        
        // Create result tensor
        let result_shape = crate::tensor::Shape::new([a_shape.dims[0], b_shape.dims[1]]);
        let result_tensor = AdvancedTensor::from_vec(result_data, result_shape)?;
        
        Ok(result_tensor)
    }
    
    /// Execute element-wise operations on GPU
    pub async fn elementwise_op<const DIMS: usize>(
        &mut self,
        a: &AdvancedTensor<f32, DIMS>,
        b: Option<&AdvancedTensor<f32, DIMS>>,
        op: KernelType,
    ) -> AnvilResult<AdvancedTensor<f32, DIMS>> {
        let size = a.size() as u32;
        let element_size = std::mem::size_of::<f32>() as u64;
        let buffer_size = size as u64 * element_size;
        
        // Create GPU buffers
        let a_buffer = self.create_buffer_from_tensor(a, "input_a")?;
        let b_buffer = if let Some(b_tensor) = b {
            self.create_buffer_from_tensor(b_tensor, "input_b")?
        } else {
            // Create dummy buffer for unary operations
            GPUBuffer::new(&self.device, buffer_size, wgpu::BufferUsages::STORAGE, "dummy_b")
        };
        
        let output_buffer = GPUBuffer::new(
            &self.device,
            buffer_size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            "output",
        );
        
        let size_buffer = self.create_uniform_buffer(&[size], "size")?;
        
        // Create bind group
        let bind_group_layout = self.create_bind_group_layout_for_kernel(op);
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("elementwise_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: b_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.buffer().as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: size_buffer.buffer().as_entire_binding(),
                },
            ],
        });
        
        // Dispatch compute shader
        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("elementwise_encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("elementwise_pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(self.pipelines.get(&op).unwrap());
            compute_pass.set_bind_group(0, &bind_group, &[]);
            
            let workgroup_size = 256;
            let dispatch_size = (size + workgroup_size - 1) / workgroup_size;
            
            compute_pass.dispatch_workgroups(dispatch_size, 1, 1);
        }
        
        let command_buffer = encoder.finish();
        self.queue.submit(std::iter::once(command_buffer));
        
        // Read back result
        let result_data: Vec<f32> = output_buffer.read_data(&self.device).await?;
        
        // Create result tensor
        let result_tensor = AdvancedTensor::from_vec(result_data, a.shape().clone())?;
        
        Ok(result_tensor)
    }
    
    fn create_buffer_from_tensor<T, const DIMS: usize>(
        &self,
        tensor: &AdvancedTensor<T, DIMS>,
        label: &str,
    ) -> AnvilResult<GPUBuffer>
    where
        T: bytemuck::Pod,
    {
        let data = tensor.as_slice::<T>();
        let size = data.len() * std::mem::size_of::<T>();
        
        let buffer = GPUBuffer::new(
            &self.device,
            size as u64,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            label,
        );
        
        buffer.write_data(&self.queue, data);
        
        Ok(buffer)
    }
    
    fn create_uniform_buffer<T>(
        &self,
        data: &[T],
        label: &str,
    ) -> AnvilResult<GPUBuffer>
    where
        T: bytemuck::Pod,
    {
        let size = data.len() * std::mem::size_of::<T>();
        let buffer = GPUBuffer::new(
            &self.device,
            size as u64,
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            label,
        );
        
        buffer.write_data(&self.queue, data);
        
        Ok(buffer)
    }
    
    /// Get device for external operations
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get queue for external operations
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
}

unsafe impl Send for ComputeEngine {}
unsafe impl Sync for ComputeEngine {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::{Shape, DType, Device};
    
    #[tokio::test]
    async fn test_gpu_matmul() {
        let mut engine = ComputeEngine::new().await.unwrap();
        
        let a = AdvancedTensor::<f32, 2>::new(
            Shape::new([4, 4]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let b = AdvancedTensor::<f32, 2>::new(
            Shape::new([4, 4]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let result = engine.matmul(&a, &b, true).await.unwrap();
        assert_eq!(result.shape().dims, [4, 4]);
    }
    
    #[tokio::test]
    async fn test_gpu_elementwise() {
        let mut engine = ComputeEngine::new().await.unwrap();
        
        let a = AdvancedTensor::<f32, 2>::new(
            Shape::new([4, 4]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let b = AdvancedTensor::<f32, 2>::new(
            Shape::new([4, 4]),
            DType::F32,
            Device::Cpu,
        ).unwrap();
        
        let result = engine.elementwise_op(&a, Some(&b), KernelType::ElementwiseAdd).await.unwrap();
        assert_eq!(result.shape().dims, [4, 4]);
    }
}