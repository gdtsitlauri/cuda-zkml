#include "nn/layers.cuh"
#include "common.cuh"

// ============================================================
// Neural Network Layer CUDA Kernels
//
// Matrix-vector multiplication (Linear layer):
//   Each thread computes one output element y[i] = sum_j W[i][j] * x[j] + b[i]
//   Uses shared memory to cache x for reuse across threads in same block.
//
// Occupancy (sm_75):
//   fp_matmul_kernel: 256 threads/block, ~40 regs/thread (Fp accumulator)
//   → 10240 regs/block → 6 blocks/SM.
//   Shared memory: min(N, 512) * 32 bytes ≤ 16KB. Allows 4 blocks/SM.
//   Effective: 4 blocks/SM × 16 SMs = 64 blocks.
//   For MNIST (N=784): 25 tiles of 32 elements = 25 iterations per output.
//
// ReLU approximation:
//   Degree-3 polynomial evaluated via Horner's method:
//   f(x) = c0 + x*(c1 + x*(c2 + x*c3))
//   3 Fp multiplications + 3 Fp additions per element.
//   Very lightweight, 256 threads/block, near-full occupancy.
// ============================================================

namespace zkml {

using bn254::Fp;

// Tile size for shared memory caching of input vector
constexpr int TILE_SIZE = 32; // 32 Fp elements = 1 KB

// ============================================================
// Matrix-vector multiplication: y[i] = sum_j W[i*N+j] * x[j] + bias[i]
// Each thread computes one output element.
// Shared memory used to cache tiles of x.
// ============================================================
__global__ void fp_matmul_kernel(Fp* y, const Fp* W, const Fp* x,
                                   const Fp* bias, int M, int N) {
    __shared__ Fp x_shared[TILE_SIZE];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;

    Fp acc = Fp::zero();

    // Process x in tiles
    for (int tile_start = 0; tile_start < N; tile_start += TILE_SIZE) {
        // Cooperatively load tile of x into shared memory
        if (threadIdx.x < TILE_SIZE && tile_start + threadIdx.x < N) {
            x_shared[threadIdx.x] = x[tile_start + threadIdx.x];
        }
        __syncthreads();

        // Compute partial dot product for this tile
        int tile_end = min(TILE_SIZE, N - tile_start);
        for (int j = 0; j < tile_end; j++) {
            acc = acc + W[row * N + tile_start + j] * x_shared[j];
        }
        __syncthreads();
    }

    // Add bias
    if (bias != nullptr) {
        acc = acc + bias[row];
    }

    y[row] = acc;
}

// ============================================================
// ReLU approximation: f(x) = c0 + x*(c1 + x*(c2 + x*c3))
// Uses Horner's method for efficient polynomial evaluation.
// All arithmetic in Fp — no comparisons, fully ZK-provable.
//
// Occupancy (sm_75):
//   256 threads/block, ~20 regs/thread (3 Fp muls + 3 Fp adds).
//   → 5120 regs/block → 12 blocks/SM. Near-full occupancy.
// ============================================================
__global__ void fp_relu_approx_kernel(Fp* out, const Fp* in,
                                        Fp c0, Fp c1, Fp c2, Fp c3,
                                        Fp scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    Fp x = in[idx];

    // Horner's method: result = c0 + x*(c1 + x*(c2 + x*c3))
    Fp result = c3;
    result = result * x + c2;
    result = result * x + c1;
    result = result * x + c0;

    // Apply rescaling
    result = result * scale;

    out[idx] = result;
}

// ============================================================
// Softmax approximation using truncated Taylor series for exp
// exp(x) ≈ 1 + x + x^2/2 + x^3/6
// Then normalize: softmax[i] = exp(x[i]) / sum(exp(x[j]))
//
// In finite field: division is multiplication by inverse.
// The "exp" here is a polynomial approximation that preserves
// relative ordering for classification tasks.
//
// Occupancy (sm_75):
//   Single block, n threads (n=10 for MNIST). Not occupancy-bound.
//   Shared memory: (n+1) * 32 bytes = 352 bytes. Negligible.
//   This kernel is latency-bound by the sequential sum reduction.
// ============================================================
__global__ void fp_softmax_approx_kernel(Fp* out, const Fp* in,
                                           Fp inv_2, Fp inv_6,
                                           int n) {
    // Phase 1: Compute approximate exp for each element
    // Use shared memory to store all exp values for normalization
    extern __shared__ Fp shared_exp[];

    int idx = threadIdx.x;
    if (idx >= n) return;

    Fp x = in[idx];

    // Taylor expansion: exp(x) ≈ 1 + x + x^2/2 + x^3/6
    Fp x2 = x * x;
    Fp x3 = x2 * x;
    Fp one = Fp::one();
    Fp exp_approx = one + x + x2 * inv_2 + x3 * inv_6;

    shared_exp[idx] = exp_approx;
    __syncthreads();

    // Phase 2: Sum all exp values (single thread)
    Fp sum = Fp::zero();
    if (idx == 0) {
        for (int i = 0; i < n; i++) {
            sum = sum + shared_exp[i];
        }
        // Store sum in shared memory for all threads to read
        shared_exp[n] = sum;
    }
    __syncthreads();

    // Phase 3: Normalize
    sum = shared_exp[n];
    Fp sum_inv = sum.inv(); // Expensive but only for small n (10 classes)
    out[idx] = exp_approx * sum_inv;
}

// ============================================================
// Host wrappers
// ============================================================

void fp_matmul_gpu(Fp* d_y, const Fp* d_W, const Fp* d_x,
                    const Fp* d_bias, int M, int N) {
    // Each thread computes one row of the output
    // Block size 256, need enough blocks for M outputs
    int threads = 256;
    int blocks = (M + threads - 1) / threads;

    // Shared memory for TILE_SIZE Fp elements
    size_t shared_mem = TILE_SIZE * sizeof(Fp);

    fp_matmul_kernel<<<blocks, threads, shared_mem>>>(d_y, d_W, d_x, d_bias, M, N);
    CUDA_CHECK(cudaGetLastError());
}

void fp_relu_approx_gpu(Fp* d_out, const Fp* d_in,
                          const ReLUApproxParams& params, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp_relu_approx_kernel<<<blocks, threads>>>(
        d_out, d_in, params.c0, params.c1, params.c2, params.c3, params.scale, n);
    CUDA_CHECK(cudaGetLastError());
}

void fp_softmax_approx_gpu(Fp* d_out, const Fp* d_in,
                             const SoftmaxApproxParams& params, int n) {
    // Single block, n+1 shared Fp elements (n for exp, 1 for sum)
    size_t shared_mem = (n + 1) * sizeof(Fp);

    // Compute 1/2 and 1/6 in Fp
    Fp inv_2 = Fp::from_uint(2).inv();
    Fp inv_6 = Fp::from_uint(6).inv();

    fp_softmax_approx_kernel<<<1, n, shared_mem>>>(d_out, d_in, inv_2, inv_6, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace zkml
