#include "curve/g1.cuh"
#include "common.cuh"

// ============================================================
// G1 CUDA Kernels
//
// Occupancy (sm_75):
//   g1_scalar_mul_kernel: ~50 registers/thread (G1 point + scalar + temp)
//   With 128 threads/block → 6400 regs/block → 10 blocks/SM.
//   GTX 1650 has 16 SMs → up to 160 active blocks.
// ============================================================

namespace bn254 {

// Parallel scalar multiplication: out[i] = scalars[i] * points[i]
__global__ void g1_scalar_mul_kernel(G1Jacobian* out,
                                       const G1Jacobian* points,
                                       const Fr* scalars,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = points[idx].scalar_mul(scalars[idx]);
    }
}

// Parallel point addition: out[i] = a[i] + b[i]
// Occupancy: 256 threads/block, ~40 regs → 10240 regs/block → 6 blocks/SM.
__global__ void g1_add_kernel(G1Jacobian* out,
                                const G1Jacobian* a,
                                const G1Jacobian* b,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// Parallel point doubling: out[i] = 2 * a[i]
// Occupancy: 256 threads/block, ~30 regs → 7680 regs/block → 8 blocks/SM.
__global__ void g1_dbl_kernel(G1Jacobian* out,
                                const G1Jacobian* a,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx].dbl();
    }
}

// Parallel mixed addition: out[i] = a[i] + b[i] (Jacobian + Affine)
// Occupancy: 256 threads/block, ~36 regs → 9216 regs/block → 7 blocks/SM.
__global__ void g1_add_mixed_kernel(G1Jacobian* out,
                                      const G1Jacobian* a,
                                      const G1Affine* b,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx].add_affine(b[idx]);
    }
}

// Batch affine conversion: convert array of Jacobian points to Affine
// Occupancy: 128 threads/block (inv is heavy, ~80 regs) → 10240 regs/block → 6 blocks/SM.
__global__ void g1_to_affine_kernel(G1Affine* out,
                                      const G1Jacobian* points,
                                      int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = points[idx].to_affine();
    }
}

// ============================================================
// Host wrappers
// ============================================================

void g1_scalar_mul_gpu(G1Jacobian* d_out, const G1Jacobian* d_points,
                        const Fr* d_scalars, int n) {
    // Use 128 threads/block for scalar mul (register-heavy)
    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    g1_scalar_mul_kernel<<<blocks, threads>>>(d_out, d_points, d_scalars, n);
    CUDA_CHECK(cudaGetLastError());
}

void g1_add_gpu(G1Jacobian* d_out, const G1Jacobian* d_a,
                 const G1Jacobian* d_b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    g1_add_kernel<<<blocks, threads>>>(d_out, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
}

void g1_to_affine_gpu(G1Affine* d_out, const G1Jacobian* d_points, int n) {
    int threads = 128; // inv is heavy
    int blocks = (n + threads - 1) / threads;
    g1_to_affine_kernel<<<blocks, threads>>>(d_out, d_points, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace bn254
