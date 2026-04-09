#include "field/fp.cuh"
#include "common.cuh"

// ============================================================
// CUDA kernels for parallel field operations
//
// These kernels operate on arrays of Fp elements, enabling
// SIMD-style parallelism for polynomial evaluation, witness
// generation, and other bulk field operations.
//
// Occupancy analysis (sm_75, GTX 1650):
//   - fp_add_kernel: ~16 regs/thread, 256 threads/block
//     → 4096 regs/block → 16 blocks/SM (max). Limited by max blocks = 16.
//   - fp_mul_kernel: ~32 regs/thread, 256 threads/block
//     → 8192 regs/block → 8 blocks/SM. Good occupancy.
//   - fp_inv_kernel: Very register-heavy due to exponentiation.
//     Use 128 threads/block → ~4 blocks/SM.
// ============================================================

namespace bn254 {

// Parallel Fp addition: out[i] = a[i] + b[i]
__global__ void fp_add_kernel(Fp* out, const Fp* a, const Fp* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

// Parallel Fp subtraction: out[i] = a[i] - b[i]
__global__ void fp_sub_kernel(Fp* out, const Fp* a, const Fp* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] - b[idx];
    }
}

// Parallel Fp multiplication: out[i] = a[i] * b[i]
__global__ void fp_mul_kernel(Fp* out, const Fp* a, const Fp* b, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx];
    }
}

// Parallel Fp scalar multiplication: out[i] = a[i] * scalar
__global__ void fp_scalar_mul_kernel(Fp* out, const Fp* a, Fp scalar, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * scalar;
    }
}

// Parallel Fp inversion: out[i] = a[i]^{-1}
// Uses Montgomery's batch inversion trick for efficiency:
// Instead of n independent inversions (each O(256) muls),
// compute all n inversions with 1 inversion + 3(n-1) muls.
__global__ void fp_inv_kernel(Fp* out, const Fp* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx].inv();
    }
}

// Montgomery batch inversion on GPU
// products[i] = a[0] * a[1] * ... * a[i]
// Invert products[n-1], then recover individual inverses
__global__ void fp_batch_inv_step1(Fp* products, const Fp* a, int n) {
    // Single thread builds prefix products (could parallelize with scan)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        products[0] = a[0];
        for (int i = 1; i < n; i++) {
            products[i] = products[i - 1] * a[i];
        }
    }
}

__global__ void fp_batch_inv_step2(Fp* out, const Fp* a, Fp* products, int n) {
    // Single thread recovers inverses (could parallelize)
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Invert the total product
        Fp inv_total = products[n - 1].inv();

        // Recover individual inverses from right to left
        for (int i = n - 1; i > 0; i--) {
            out[i] = products[i - 1] * inv_total;
            inv_total = inv_total * a[i];
        }
        out[0] = inv_total;
    }
}

// Parallel Fp squaring: out[i] = a[i]^2
__global__ void fp_sqr_kernel(Fp* out, const Fp* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx].sqr();
    }
}

// Parallel Fp negation: out[i] = -a[i]
__global__ void fp_neg_kernel(Fp* out, const Fp* a, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = -a[idx];
    }
}

// ============================================================
// Host-side wrappers
// ============================================================

void fp_add_gpu(Fp* d_out, const Fp* d_a, const Fp* d_b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp_add_kernel<<<blocks, threads>>>(d_out, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
}

void fp_sub_gpu(Fp* d_out, const Fp* d_a, const Fp* d_b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp_sub_kernel<<<blocks, threads>>>(d_out, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
}

void fp_mul_gpu(Fp* d_out, const Fp* d_a, const Fp* d_b, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fp_mul_kernel<<<blocks, threads>>>(d_out, d_a, d_b, n);
    CUDA_CHECK(cudaGetLastError());
}

void fp_batch_inv_gpu(Fp* d_out, const Fp* d_a, int n) {
    Fp* d_products;
    CUDA_CHECK(cudaMalloc(&d_products, n * sizeof(Fp)));
    fp_batch_inv_step1<<<1, 1>>>(d_products, d_a, n);
    CUDA_CHECK(cudaGetLastError());
    fp_batch_inv_step2<<<1, 1>>>(d_out, d_a, d_products, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaFree(d_products));
}

} // namespace bn254
