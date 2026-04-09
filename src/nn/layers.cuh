#pragma once

#include "field/fp.cuh"
#include "nn/quantize.cuh"

// ============================================================
// Neural Network Layers in Finite Field Arithmetic
//
// All operations happen in Fp (BN254 base field).
// This allows every intermediate value to be used in ZK proofs.
//
// Supported layers:
// 1. Linear (Dense): y = Wx + b  (matrix multiply in Fp)
// 2. ReLU approximation: polynomial approximation in Fp
// 3. Softmax approximation: polynomial approximation in Fp
// 4. Self-attention: tiny single-head transformer block over flattened tokens
//
// ReLU approximation strategy:
//   ReLU(x) = max(0, x) ≈ polynomial in Fp
//   We use a degree-3 polynomial: f(x) = a0 + a1*x + a2*x^2 + a3*x^3
//   Fitted to approximate ReLU over [-128, 127] (int8 range)
//   Coefficients precomputed and stored as Fp elements.
//
// For ZK provability, every operation must be an R1CS-compatible
// arithmetic gate (add, mul). No comparisons or branching.
//
// Memory: For MNIST MLP (784→128→10):
//   Weights: 784*128 + 128*10 = 101,632 Fp elements = 3.2 MB
//   Activations: 784 + 128 + 10 = 922 Fp elements = 29 KB
//   Total << 3.5 GB VRAM budget
// ============================================================

namespace zkml {

// Layer types
enum class LayerType {
    LINEAR,
    RELU_APPROX,
    SOFTMAX_APPROX,
    SELF_ATTENTION
};

// Linear layer: y = Wx + b
struct LinearLayer {
    bn254::Fp* weights;  // [out_features × in_features], row-major
    bn254::Fp* bias;     // [out_features]
    int in_features;
    int out_features;
};

// ReLU approximation coefficients (degree 3)
// f(x) = c0 + c1*x + c2*x^2 + c3*x^3
// Approximates max(0, x) for x in quantized range
struct ReLUApproxParams {
    bn254::Fp c0, c1, c2, c3;
    bn254::Fp scale; // rescaling factor after polynomial
};

// Softmax approximation using exp Taylor series
// exp(x) ≈ 1 + x + x^2/2 + x^3/6 (degree 3)
struct SoftmaxApproxParams {
    int num_classes;
    bn254::Fp inv_scale; // 1/temperature
};

// Self-attention over a flattened [seq_len x hidden_size] input.
struct SelfAttentionParams {
    int seq_len;
    int hidden_size;
    int num_heads;
    bn254::Fp inv_scale; // scaling factor applied to attention scores
};

// Layer descriptor
struct LayerDesc {
    LayerType type;
    int in_size;
    int out_size;
    union {
        struct { int in_features; int out_features; } linear;
        struct { int size; } relu;
        struct { int num_classes; } softmax;
        struct { int seq_len; int hidden_size; int num_heads; } attention;
    };
};

// ============================================================
// CUDA kernel declarations
// ============================================================

// Matrix-vector multiply in Fp: y = W * x + b
// W is [M x N], x is [N], y is [M]
void fp_matmul_gpu(bn254::Fp* d_y, const bn254::Fp* d_W,
                    const bn254::Fp* d_x, const bn254::Fp* d_bias,
                    int M, int N);

// ReLU approximation in Fp
void fp_relu_approx_gpu(bn254::Fp* d_out, const bn254::Fp* d_in,
                          const ReLUApproxParams& params, int n);

// Softmax approximation in Fp
void fp_softmax_approx_gpu(bn254::Fp* d_out, const bn254::Fp* d_in,
                             const SoftmaxApproxParams& params, int n);

} // namespace zkml
