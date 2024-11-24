// src/cosine_similarity.wgsl

// Uniforms
struct Params {
    vector_length: u32,
    num_vectors: u32,
};
@group(0) @binding(3)
var<uniform> params: Params;

// Buffers
// Binding 0: Target vector
@group(0) @binding(0)
var<storage, read> target_vector: array<f32>;

// Binding 1: Vectors buffer
@group(0) @binding(1)
var<storage, read> vectors: array<f32>;

// Binding 2: Output buffer
@group(0) @binding(2)
var<storage, read_write> output: array<f32>;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) GlobalInvocationID: vec3<u32>) {
    let vector_index = GlobalInvocationID.x;

    if (vector_index >= params.num_vectors) {
        return;
    }

    var dot_product: f32 = 0.0;
    var target_norm: f32 = 0.0;
    var vector_norm: f32 = 0.0;

    for (var i: u32 = 0u; i < params.vector_length; i = i + 1u) {
        let target_value = target_vector[i];
        let vector_value = vectors[vector_index * params.vector_length + i];
        dot_product = dot_product + target_value * vector_value;
        target_norm = target_norm + target_value * target_value;
        vector_norm = vector_norm + vector_value * vector_value;
    }

    target_norm = sqrt(target_norm);
    vector_norm = sqrt(vector_norm);

    var cosine_similarity: f32 = 0.0;

    if (target_norm > 0.0 && vector_norm > 0.0) {
        cosine_similarity = dot_product / (target_norm * vector_norm);
    }

    output[vector_index] = cosine_similarity;
}
