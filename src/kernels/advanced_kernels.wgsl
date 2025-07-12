// Advanced GPU compute shaders for Anvil ML Framework
// Optimized kernels for matrix operations, convolutions, and neural network layers

// ============================================================================
// MATRIX MULTIPLICATION KERNELS
// ============================================================================

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;
@group(0) @binding(3) var<uniform> dimensions: vec4<u32>; // m, k, n, batch_size

// Workgroup shared memory for tiling optimization
var<workgroup> tile_a: array<array<f32, 16>, 16>;
var<workgroup> tile_b: array<array<f32, 16>, 16>;

const TILE_SIZE: u32 = 16u;

// Optimized tiled matrix multiplication with shared memory
@compute @workgroup_size(16, 16)
fn matmul_tiled(@builtin(global_invocation_id) global_id: vec3<u32>,
                @builtin(local_invocation_id) local_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let local_row = local_id.x;
    let local_col = local_id.y;
    
    let m = dimensions.x;
    let k = dimensions.y;
    let n = dimensions.z;
    
    var sum = 0.0;
    
    // Iterate over tiles
    for (var tile = 0u; tile < (k + TILE_SIZE - 1u) / TILE_SIZE; tile++) {
        // Load tiles into shared memory
        let tile_row = tile * TILE_SIZE + local_col;
        let tile_col = tile * TILE_SIZE + local_row;
        
        // Load tile from matrix A
        if (row < m && tile_row < k) {
            tile_a[local_row][local_col] = a[row * k + tile_row];
        } else {
            tile_a[local_row][local_col] = 0.0;
        }
        
        // Load tile from matrix B
        if (tile_col < k && col < n) {
            tile_b[local_row][local_col] = b[tile_col * n + col];
        } else {
            tile_b[local_row][local_col] = 0.0;
        }
        
        // Synchronize workgroup
        workgroupBarrier();
        
        // Compute partial dot product
        for (var i = 0u; i < TILE_SIZE; i++) {
            sum += tile_a[local_row][i] * tile_b[i][local_col];
        }
        
        // Synchronize before loading next tile
        workgroupBarrier();
    }
    
    // Write result
    if (row < m && col < n) {
        c[row * n + col] = sum;
    }
}

// Batched matrix multiplication for training efficiency
@compute @workgroup_size(16, 16)
fn matmul_batched(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let batch = global_id.z;
    
    let m = dimensions.x;
    let k = dimensions.y;
    let n = dimensions.z;
    let batch_size = dimensions.w;
    
    if (row >= m || col >= n || batch >= batch_size) {
        return;
    }
    
    let batch_offset_a = batch * m * k;
    let batch_offset_b = batch * k * n;
    let batch_offset_c = batch * m * n;
    
    var sum = 0.0;
    for (var i = 0u; i < k; i++) {
        let a_idx = batch_offset_a + row * k + i;
        let b_idx = batch_offset_b + i * n + col;
        sum += a[a_idx] * b[b_idx];
    }
    
    let c_idx = batch_offset_c + row * n + col;
    c[c_idx] = sum;
}

// ============================================================================
// ELEMENT-WISE OPERATIONS
// ============================================================================

@group(1) @binding(0) var<storage, read> input_a: array<f32>;
@group(1) @binding(1) var<storage, read> input_b: array<f32>;
@group(1) @binding(2) var<storage, read_write> output: array<f32>;
@group(1) @binding(3) var<uniform> size: u32;

@compute @workgroup_size(256)
fn elementwise_add(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= size) {
        return;
    }
    output[index] = input_a[index] + input_b[index];
}

@compute @workgroup_size(256)
fn elementwise_mul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= size) {
        return;
    }
    output[index] = input_a[index] * input_b[index];
}

@compute @workgroup_size(256)
fn elementwise_relu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= size) {
        return;
    }
    output[index] = max(0.0, input_a[index]);
}

@compute @workgroup_size(256)
fn elementwise_gelu(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= size) {
        return;
    }
    let x = input_a[index];
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
    let x_cubed = x * x * x;
    let inner = 0.7978845608028654 * (x + 0.044715 * x_cubed); // sqrt(2/π)
    output[index] = 0.5 * x * (1.0 + tanh(inner));
}

@compute @workgroup_size(256)
fn elementwise_swish(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= size) {
        return;
    }
    let x = input_a[index];
    output[index] = x / (1.0 + exp(-x)); // x * sigmoid(x)
}

// ============================================================================
// REDUCTION OPERATIONS
// ============================================================================

@group(2) @binding(0) var<storage, read> input: array<f32>;
@group(2) @binding(1) var<storage, read_write> output_reduction: array<f32>;
@group(2) @binding(2) var<uniform> reduction_size: u32;

var<workgroup> shared_data: array<f32, 256>;

@compute @workgroup_size(256)
fn reduce_sum(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Load data into shared memory
    if (gid < reduction_size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = 0.0;
    }
    
    workgroupBarrier();
    
    // Perform reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride && tid + stride < 256u) {
            shared_data[tid] += shared_data[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result
    if (tid == 0u) {
        output_reduction[workgroup_id.x] = shared_data[0];
    }
}

@compute @workgroup_size(256)
fn reduce_max(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_id.x;
    let gid = global_id.x;
    
    // Load data into shared memory
    if (gid < reduction_size) {
        shared_data[tid] = input[gid];
    } else {
        shared_data[tid] = -3.40282347e+38; // -FLT_MAX
    }
    
    workgroupBarrier();
    
    // Perform reduction in shared memory
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride && tid + stride < 256u) {
            shared_data[tid] = max(shared_data[tid], shared_data[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    // Write result
    if (tid == 0u) {
        output_reduction[workgroup_id.x] = shared_data[0];
    }
}

// ============================================================================
// CONVOLUTION KERNELS
// ============================================================================

@group(3) @binding(0) var<storage, read> conv_input: array<f32>;
@group(3) @binding(1) var<storage, read> conv_weight: array<f32>;
@group(3) @binding(2) var<storage, read> conv_bias: array<f32>;
@group(3) @binding(3) var<storage, read_write> conv_output: array<f32>;
@group(3) @binding(4) var<uniform> conv_params: vec4<u32>; // in_channels, out_channels, kernel_size, stride
@group(3) @binding(5) var<uniform> conv_dims: vec4<u32>; // batch_size, height, width, padding

@compute @workgroup_size(16, 16)
fn conv2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_y = global_id.x;
    let out_x = global_id.y;
    let out_c = global_id.z;
    
    let in_channels = conv_params.x;
    let out_channels = conv_params.y;
    let kernel_size = conv_params.z;
    let stride = conv_params.w;
    
    let batch_size = conv_dims.x;
    let height = conv_dims.y;
    let width = conv_dims.z;
    let padding = conv_dims.w;
    
    let out_height = (height + 2u * padding - kernel_size) / stride + 1u;
    let out_width = (width + 2u * padding - kernel_size) / stride + 1u;
    
    if (out_y >= out_height || out_x >= out_width || out_c >= out_channels) {
        return;
    }
    
    for (var batch = 0u; batch < batch_size; batch++) {
        var sum = 0.0;
        
        // Apply convolution
        for (var in_c = 0u; in_c < in_channels; in_c++) {
            for (var ky = 0u; ky < kernel_size; ky++) {
                for (var kx = 0u; kx < kernel_size; kx++) {
                    let in_y = out_y * stride + ky;
                    let in_x = out_x * stride + kx;
                    
                    // Check bounds with padding
                    if (in_y >= padding && in_y < height + padding &&
                        in_x >= padding && in_x < width + padding) {
                        let actual_y = in_y - padding;
                        let actual_x = in_x - padding;
                        
                        if (actual_y < height && actual_x < width) {
                            let input_idx = batch * in_channels * height * width +
                                          in_c * height * width +
                                          actual_y * width + actual_x;
                            
                            let weight_idx = out_c * in_channels * kernel_size * kernel_size +
                                           in_c * kernel_size * kernel_size +
                                           ky * kernel_size + kx;
                            
                            sum += conv_input[input_idx] * conv_weight[weight_idx];
                        }
                    }
                }
            }
        }
        
        // Add bias and write output
        let output_idx = batch * out_channels * out_height * out_width +
                        out_c * out_height * out_width +
                        out_y * out_width + out_x;
        
        conv_output[output_idx] = sum + conv_bias[out_c];
    }
}

// Depthwise separable convolution for efficiency
@compute @workgroup_size(16, 16)
fn depthwise_conv2d(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_y = global_id.x;
    let out_x = global_id.y;
    let channel = global_id.z;
    
    let kernel_size = conv_params.z;
    let stride = conv_params.w;
    let batch_size = conv_dims.x;
    let height = conv_dims.y;
    let width = conv_dims.z;
    let padding = conv_dims.w;
    
    let out_height = (height + 2u * padding - kernel_size) / stride + 1u;
    let out_width = (width + 2u * padding - kernel_size) / stride + 1u;
    
    if (out_y >= out_height || out_x >= out_width || channel >= conv_params.x) {
        return;
    }
    
    for (var batch = 0u; batch < batch_size; batch++) {
        var sum = 0.0;
        
        // Depthwise convolution (each channel processed independently)
        for (var ky = 0u; ky < kernel_size; ky++) {
            for (var kx = 0u; kx < kernel_size; kx++) {
                let in_y = out_y * stride + ky;
                let in_x = out_x * stride + kx;
                
                if (in_y >= padding && in_y < height + padding &&
                    in_x >= padding && in_x < width + padding) {
                    let actual_y = in_y - padding;
                    let actual_x = in_x - padding;
                    
                    if (actual_y < height && actual_x < width) {
                        let input_idx = batch * conv_params.x * height * width +
                                      channel * height * width +
                                      actual_y * width + actual_x;
                        
                        let weight_idx = channel * kernel_size * kernel_size +
                                       ky * kernel_size + kx;
                        
                        sum += conv_input[input_idx] * conv_weight[weight_idx];
                    }
                }
            }
        }
        
        let output_idx = batch * conv_params.x * out_height * out_width +
                        channel * out_height * out_width +
                        out_y * out_width + out_x;
        
        conv_output[output_idx] = sum + conv_bias[channel];
    }
}

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

@group(4) @binding(0) var<storage, read> softmax_input: array<f32>;
@group(4) @binding(1) var<storage, read_write> softmax_output: array<f32>;
@group(4) @binding(2) var<uniform> softmax_dims: vec2<u32>; // batch_size, feature_size

var<workgroup> max_vals: array<f32, 256>;
var<workgroup> sum_vals: array<f32, 256>;

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) global_id: vec3<u32>,
           @builtin(local_invocation_id) local_id: vec3<u32>,
           @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let batch = workgroup_id.x;
    let tid = local_id.x;
    let feature_size = softmax_dims.y;
    
    let batch_offset = batch * feature_size;
    
    // Phase 1: Find maximum value for numerical stability
    var local_max = -3.40282347e+38; // -FLT_MAX
    for (var i = tid; i < feature_size; i += 256u) {
        local_max = max(local_max, softmax_input[batch_offset + i]);
    }
    max_vals[tid] = local_max;
    
    workgroupBarrier();
    
    // Reduce to find global maximum
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            max_vals[tid] = max(max_vals[tid], max_vals[tid + stride]);
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    let global_max = max_vals[0];
    workgroupBarrier();
    
    // Phase 2: Compute sum of exponentials
    var local_sum = 0.0;
    for (var i = tid; i < feature_size; i += 256u) {
        local_sum += exp(softmax_input[batch_offset + i] - global_max);
    }
    sum_vals[tid] = local_sum;
    
    workgroupBarrier();
    
    // Reduce to find global sum
    stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            sum_vals[tid] += sum_vals[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    let global_sum = sum_vals[0];
    workgroupBarrier();
    
    // Phase 3: Compute final softmax values
    for (var i = tid; i < feature_size; i += 256u) {
        let exp_val = exp(softmax_input[batch_offset + i] - global_max);
        softmax_output[batch_offset + i] = exp_val / global_sum;
    }
}

// ============================================================================
// ATTENTION MECHANISMS
// ============================================================================

@group(5) @binding(0) var<storage, read> queries: array<f32>;
@group(5) @binding(1) var<storage, read> keys: array<f32>;
@group(5) @binding(2) var<storage, read> values: array<f32>;
@group(5) @binding(3) var<storage, read_write> attention_output: array<f32>;
@group(5) @binding(4) var<uniform> attention_dims: vec4<u32>; // batch_size, seq_len, num_heads, head_dim

var<workgroup> attention_scores: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn multi_head_attention(@builtin(global_invocation_id) global_id: vec3<u32>,
                        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let batch = global_id.z;
    let head = global_id.y / 16u; // Assuming 16 is our tile size
    let seq_pos = global_id.x;
    
    let batch_size = attention_dims.x;
    let seq_len = attention_dims.y;
    let num_heads = attention_dims.z;
    let head_dim = attention_dims.w;
    
    if (batch >= batch_size || head >= num_heads || seq_pos >= seq_len) {
        return;
    }
    
    let scale = 1.0 / sqrt(f32(head_dim));
    
    // Compute attention scores for this query position
    for (var key_pos = 0u; key_pos < seq_len; key_pos++) {
        var score = 0.0;
        
        // Dot product between query and key
        for (var d = 0u; d < head_dim; d++) {
            let q_idx = batch * num_heads * seq_len * head_dim +
                       head * seq_len * head_dim +
                       seq_pos * head_dim + d;
            
            let k_idx = batch * num_heads * seq_len * head_dim +
                       head * seq_len * head_dim +
                       key_pos * head_dim + d;
            
            score += queries[q_idx] * keys[k_idx];
        }
        
        // Scale and store
        attention_scores[key_pos] = score * scale;
    }
    
    // Apply softmax to attention scores
    // Find max for numerical stability
    var max_score = -3.40282347e+38;
    for (var i = 0u; i < seq_len; i++) {
        max_score = max(max_score, attention_scores[i]);
    }
    
    // Compute exp and sum
    var sum_exp = 0.0;
    for (var i = 0u; i < seq_len; i++) {
        attention_scores[i] = exp(attention_scores[i] - max_score);
        sum_exp += attention_scores[i];
    }
    
    // Normalize
    for (var i = 0u; i < seq_len; i++) {
        attention_scores[i] /= sum_exp;
    }
    
    // Compute weighted sum of values
    for (var d = 0u; d < head_dim; d++) {
        var weighted_sum = 0.0;
        
        for (var val_pos = 0u; val_pos < seq_len; val_pos++) {
            let v_idx = batch * num_heads * seq_len * head_dim +
                       head * seq_len * head_dim +
                       val_pos * head_dim + d;
            
            weighted_sum += attention_scores[val_pos] * values[v_idx];
        }
        
        let out_idx = batch * num_heads * seq_len * head_dim +
                     head * seq_len * head_dim +
                     seq_pos * head_dim + d;
        
        attention_output[out_idx] = weighted_sum;
    }
}

// ============================================================================
// NORMALIZATION LAYERS
// ============================================================================

@group(6) @binding(0) var<storage, read> norm_input: array<f32>;
@group(6) @binding(1) var<storage, read> norm_weight: array<f32>;
@group(6) @binding(2) var<storage, read> norm_bias: array<f32>;
@group(6) @binding(3) var<storage, read_write> norm_output: array<f32>;
@group(6) @binding(4) var<uniform> norm_dims: vec4<u32>; // batch_size, channels, height, width

var<workgroup> mean_vals: array<f32, 256>;
var<workgroup> var_vals: array<f32, 256>;

@compute @workgroup_size(256)
fn layer_norm(@builtin(global_invocation_id) global_id: vec3<u32>,
              @builtin(local_invocation_id) local_id: vec3<u32>,
              @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let batch = workgroup_id.x;
    let tid = local_id.x;
    let feature_size = norm_dims.y * norm_dims.z * norm_dims.w;
    
    let batch_offset = batch * feature_size;
    let eps = 1e-5;
    
    // Phase 1: Compute mean
    var local_sum = 0.0;
    for (var i = tid; i < feature_size; i += 256u) {
        local_sum += norm_input[batch_offset + i];
    }
    mean_vals[tid] = local_sum;
    
    workgroupBarrier();
    
    // Reduce to find mean
    var stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            mean_vals[tid] += mean_vals[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    let mean = mean_vals[0] / f32(feature_size);
    workgroupBarrier();
    
    // Phase 2: Compute variance
    var local_var = 0.0;
    for (var i = tid; i < feature_size; i += 256u) {
        let diff = norm_input[batch_offset + i] - mean;
        local_var += diff * diff;
    }
    var_vals[tid] = local_var;
    
    workgroupBarrier();
    
    // Reduce to find variance
    stride = 128u;
    while (stride > 0u) {
        if (tid < stride) {
            var_vals[tid] += var_vals[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }
    
    let variance = var_vals[0] / f32(feature_size);
    let std_dev = sqrt(variance + eps);
    workgroupBarrier();
    
    // Phase 3: Normalize and scale
    for (var i = tid; i < feature_size; i += 256u) {
        let normalized = (norm_input[batch_offset + i] - mean) / std_dev;
        norm_output[batch_offset + i] = normalized * norm_weight[i] + norm_bias[i];
    }
}