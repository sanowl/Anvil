@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> c: array<f32>;

@group(0) @binding(3) var<uniform> dimensions: vec4<u32>; // m, k, n, stride

@compute @workgroup_size(16, 16)
fn matmul(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let m = dimensions.x;
    let k = dimensions.y;
    let n = dimensions.z;
    
    if (row >= m || col >= n) {
        return;
    }
    
    var sum = 0.0;
    for (var i = 0u; i < k; i++) {
        sum += a[row * k + i] * b[i * n + col];
    }
    c[row * n + col] = sum;
} 