#include "curve/g2.cuh"
#include "common.cuh"

// ============================================================
// G2 CUDA Kernels
//
// Occupancy (sm_75):
//   G2 operations use Fp2 arithmetic (~2x Fp cost).
//   g2_scalar_mul_kernel: ~100 registers/thread.
//   With 64 threads/block → 6400 regs/block → 10 blocks/SM.
// ============================================================

namespace bn254 {

__global__ void g2_scalar_mul_kernel(G2Jacobian* out,
                                       const G2Jacobian* points,
                                       const Fr* scalars,
                                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = points[idx].scalar_mul(scalars[idx]);
    }
}

// Occupancy: 64 threads/block, ~100 regs → 6400 regs/block → 10 blocks/SM.
__global__ void g2_add_kernel(G2Jacobian* out,
                                const G2Jacobian* a,
                                const G2Jacobian* b,
                                int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

void g2_scalar_mul_gpu(G2Jacobian* d_out, const G2Jacobian* d_points,
                        const Fr* d_scalars, int n) {
    int threads = 64; // G2 is register-heavy
    int blocks = (n + threads - 1) / threads;
    g2_scalar_mul_kernel<<<blocks, threads>>>(d_out, d_points, d_scalars, n);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace bn254
