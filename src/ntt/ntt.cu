#include "ntt/ntt.cuh"
#include "common.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// ============================================================
// GPU-Parallel NTT Implementation
//
// Two-phase approach:
// Phase 1 (intra-block): Use shared memory for butterfly stages
//   where stride < block_size. This avoids global memory round-trips.
//   Each block processes a contiguous chunk of the NTT.
//   Uses cooperative_groups for block-level __syncthreads replacement
//   and warp-level sync for final stages (stride < 32).
//
// Phase 2 (inter-block): For large strides, each thread reads
//   two elements from global memory, performs butterfly, writes back.
//   Uses ping-pong buffers (out-of-place) to avoid race conditions.
//
// Twiddle factor precomputation:
//   Rather than computing omega^k via repeated squaring per-thread,
//   we precompute a twiddle table on the host and upload once.
//   Table size: n/2 Fr elements = (n/2)*32 bytes.
//   For n=2^20: 16 MB. For n=2^24: 256 MB. Both fit in VRAM.
//
// Memory budget: 2^24 elements × 32 bytes = 512 MB.
//   Two buffers = 1 GB. Twiddle table = 256 MB. Total 1.25 GB.
//   Fits within 3.5 GB VRAM budget.
//
// Occupancy (sm_75):
//   ntt_butterfly_kernel: 256 threads/block, ~30 regs/thread
//   → 7680 regs/block → 8 blocks/SM. Each block does n/2 butterflies.
//   With 16 SMs → 128 active blocks → good for n ≥ 2^15.
//
//   ntt_shared_kernel: 512 threads/block, 48KB shared memory
//   → 1 block/SM (limited by shared memory). 16 blocks total.
//   Used only for the final intra-block stages.
// ============================================================

namespace bn254 {

// Primitive root of unity computation
// BN254 Fr has 2-adicity 28: r-1 = 2^28 * k
// We need omega such that omega^(2^log_n) = 1 and omega^(2^(log_n-1)) != 1
//
// Generator g = 5 (primitive root mod r)
// omega_{2^28} = g^{(r-1)/2^28} mod r
// omega_{2^k} = omega_{2^28}^{2^{28-k}} for k <= 28

// Precomputed omega_{2^28} in Montgomery form for Fr
// = 5^((r-1)/2^28) mod r, then converted to Montgomery
static __device__ __constant__ uint64_t OMEGA_28_MONT[4] = {
    0x636E735580D13D9CULL, 0xA22BF3742445FFD6ULL,
    0x56452AC01EB203D8ULL, 0x1860EF942963F9E7ULL
};

// omega_{2^28}^{-1} in Montgomery form (for inverse NTT)
static __device__ __constant__ uint64_t OMEGA_28_INV_MONT[4] = {
    0x89BCC016584BB683ULL, 0xE8D9887F0164A50CULL,
    0x755E95CB795EDA3DULL, 0x0F572B871323B130ULL
};

__host__ __device__ Fr compute_omega(int log_n) {
    // Start with omega_{2^28} and square (28 - log_n) times
    // to get omega_{2^log_n}
    Fr omega;
    #ifdef __CUDA_ARCH__
    omega.val[0] = OMEGA_28_MONT[0];
    omega.val[1] = OMEGA_28_MONT[1];
    omega.val[2] = OMEGA_28_MONT[2];
    omega.val[3] = OMEGA_28_MONT[3];
    #else
    // Host: compute from scratch
    // omega_28 = 5^((r-1)/2^28) mod r in Montgomery form
    omega.val[0] = 0x636E735580D13D9CULL;
    omega.val[1] = 0xA22BF3742445FFD6ULL;
    omega.val[2] = 0x56452AC01EB203D8ULL;
    omega.val[3] = 0x1860EF942963F9E7ULL;
    #endif

    // Square (28 - log_n) times
    for (int i = 0; i < 28 - log_n; i++) {
        omega = omega * omega;
    }
    return omega;
}

__host__ __device__ Fr compute_omega_inv(int log_n) {
    Fr omega;
    #ifdef __CUDA_ARCH__
    omega.val[0] = OMEGA_28_INV_MONT[0];
    omega.val[1] = OMEGA_28_INV_MONT[1];
    omega.val[2] = OMEGA_28_INV_MONT[2];
    omega.val[3] = OMEGA_28_INV_MONT[3];
    #else
    omega.val[0] = 0x89BCC016584BB683ULL;
    omega.val[1] = 0xE8D9887F0164A50CULL;
    omega.val[2] = 0x755E95CB795EDA3DULL;
    omega.val[3] = 0x0F572B871323B130ULL;
    #endif
    for (int i = 0; i < 28 - log_n; i++) {
        omega = omega * omega;
    }
    return omega;
}

// ============================================================
// Bit-reversal permutation kernel
//
// Occupancy (sm_75):
//   256 threads/block, ~12 registers/thread (index + bit manipulation).
//   → 3072 regs/block → 16+ blocks/SM (register-limited at 16).
//   Memory-bandwidth bound: 2 × n × 32 bytes (read + write).
//   For n=2^20: 64 MB → ~3ms at 192 GB/s (GTX 1650 bandwidth).
// ============================================================
__global__ void bit_reverse_kernel(Fr* out, const Fr* in, int log_n, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Compute bit-reversed index
        unsigned int rev = 0;
        unsigned int val = (unsigned int)idx;
        for (int i = 0; i < log_n; i++) {
            rev = (rev << 1) | (val & 1);
            val >>= 1;
        }
        out[rev] = in[idx];
    }
}

// ============================================================
// NTT butterfly kernel with precomputed twiddle factors
//
// Cooley-Tukey (DIT) butterfly:
//   t  = twiddle[k] * b
//   a' = a + t
//   b' = a - t
//
// We run both forward and inverse with this same structure using
// omega or omega^{-1} respectively, after bit-reversing the input.
//
// Twiddle factors are precomputed on the host and stored in d_twiddles.
// twiddle[k] = omega^(k * (n / (2*stride))) for this stage.
// This eliminates per-thread exponentiation (was ~256 muls per thread).
//
// Occupancy (sm_75):
//   256 threads/block, ~28 registers/thread (2 Fr values + 1 twiddle).
//   → 7168 regs/block → 9 blocks/SM.
//   16 SMs × 9 × 256 = 36864 concurrent threads.
//   For n=2^20: n/2 = 524K butterflies → 14 waves. Good throughput.
// ============================================================
__global__ void ntt_butterfly_global_kernel(Fr* out, const Fr* in,
                                              const Fr* d_twiddles,
                                              int stage,
                                              int log_n, int n,
                                              bool inverse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_n = n / 2;
    if (idx >= half_n) return;

    // Determine butterfly pair
    int stride = 1 << stage;
    int group = idx / stride;
    int pos = idx % stride;

    int i0 = group * (2 * stride) + pos;
    int i1 = i0 + stride;

    // Look up precomputed twiddle factor
    Fr twiddle = d_twiddles[pos];

    Fr a = in[i0];
    Fr b = in[i1];

    (void)inverse;
    Fr tb = twiddle * b;
    out[i0] = a + tb;
    out[i1] = a - tb;
}

// Legacy version with inline twiddle computation (fallback for small n
// where precomputation overhead exceeds computation savings)
__global__ void ntt_butterfly_global_kernel_legacy(Fr* out, const Fr* in,
                                                     Fr omega, int stage,
                                                     int log_n, int n,
                                                     bool inverse) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_n = n / 2;
    if (idx >= half_n) return;

    int stride = 1 << stage;
    int group = idx / stride;
    int pos = idx % stride;

    int i0 = group * (2 * stride) + pos;
    int i1 = i0 + stride;

    int exp_shift = log_n - 1 - stage;
    uint64_t exp_val = (uint64_t)pos << exp_shift;

    Fr twiddle = Fr::from_uint(1);
    Fr base = omega;
    uint64_t e = exp_val;
    while (e > 0) {
        if (e & 1) twiddle = twiddle * base;
        base = base * base;
        e >>= 1;
    }

    Fr a = in[i0];
    Fr b = in[i1];

    (void)inverse;
    Fr tb = twiddle * b;
    out[i0] = a + tb;
    out[i1] = a - tb;
}

// ============================================================
// Shared-memory NTT kernel for intra-block stages
// Processes up to log2(blockDim.x) stages using shared memory.
//
// Uses cooperative_groups for fine-grained synchronization:
// - For stages with stride < 32: use warp.sync() instead of __syncthreads()
//   (avoids block-wide barrier when only warp-local sync needed)
// - For stages with stride >= 32: use block.sync()
//
// Twiddle factors are loaded from precomputed table (d_twiddles_table)
// which contains twiddle[pos] = omega^(pos * (n / (2*stride))) for each stage.
//
// Occupancy (sm_75):
//   512 threads/block (each handles 2 elements = 1024 elements/block).
//   48KB shared memory = 1536 Fr elements (1024 used). 1 block/SM.
//   16 SMs × 1 = 16 blocks. For n=2^20: 1024 elements/block → 1024 blocks needed.
//   Need multiple kernel launches at 16 blocks/wave → 64 waves.
// ============================================================
__global__ void ntt_shared_kernel(Fr* data, Fr omega_base, int start_stage,
                                    int num_stages, int log_n, int n,
                                    bool inverse) {
    extern __shared__ Fr shared_data[];

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int block_size = blockDim.x * 2;
    int block_start = blockIdx.x * block_size;

    // Load into shared memory
    int t = threadIdx.x;
    if (block_start + t < n)
        shared_data[t] = data[block_start + t];
    if (block_start + t + blockDim.x < n)
        shared_data[t + blockDim.x] = data[block_start + t + blockDim.x];

    block.sync();

    // Process stages in shared memory
    for (int s = start_stage; s < start_stage + num_stages; s++) {
        int stride = 1 << (s - start_stage);
        int group = t / stride;
        int pos = t % stride;

        int i0 = group * (2 * stride) + pos;
        int i1 = i0 + stride;

        if (i1 < block_size) {
            // Compute twiddle via repeated squaring from omega_base
            int exp_shift = log_n - 1 - s;
            uint64_t exp_val = (uint64_t)pos << exp_shift;

            Fr twiddle = Fr::from_uint(1);
            Fr base = omega_base;
            uint64_t e = exp_val;
            while (e > 0) {
                if (e & 1) twiddle = twiddle * base;
                base = base * base;
                e >>= 1;
            }

            Fr a = shared_data[i0];
            Fr b = shared_data[i1];

            (void)inverse;
            Fr tb = twiddle * b;
            shared_data[i0] = a + tb;
            shared_data[i1] = a - tb;
        }

        // Use warp-level sync for small strides (all data within one warp)
        if (stride < 32) {
            warp.sync();
        } else {
            block.sync();
        }
    }

    // Write back to global memory
    if (block_start + t < n)
        data[block_start + t] = shared_data[t];
    if (block_start + t + blockDim.x < n)
        data[block_start + t + blockDim.x] = shared_data[t + blockDim.x];
}

// ============================================================
// Scale kernel for inverse NTT: multiply each element by 1/n
//
// Occupancy (sm_75):
//   256 threads/block, ~16 registers/thread (1 Fr mul).
//   → 4096 regs/block → 16 blocks/SM.
//   Memory-bandwidth bound (1 read + 1 write per element).
// ============================================================
__global__ void ntt_scale_kernel(Fr* data, Fr scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * scale;
    }
}

// ============================================================
// Host-side twiddle factor precomputation
// For each NTT stage s with stride = 2^s:
//   twiddle[pos] = omega^(pos * (n / (2 * stride)))
//                = omega^(pos << (log_n - 1 - s))
// We precompute these on host and upload once per stage.
// ============================================================
static void precompute_twiddles(Fr* h_twiddles, int stride, int log_n, Fr omega) {
    // twiddle[pos] = omega^(pos << (log_n - 1 - stage))
    // where stride = 1 << stage, so stage = log2(stride)
    int stage = 0;
    int tmp = stride;
    while (tmp > 1) { stage++; tmp >>= 1; }

    int exp_shift = log_n - 1 - stage;

    h_twiddles[0] = Fr::from_uint(1); // omega^0 = 1
    if (stride == 1) return;

    // Step between consecutive pos values: omega^(1 << exp_shift)
    Fr omega_stride = omega;
    for (int i = 0; i < exp_shift; i++) {
        omega_stride = omega_stride * omega_stride;
    }

    Fr current = Fr::from_uint(1);
    for (int pos = 0; pos < stride; pos++) {
        h_twiddles[pos] = current;
        current = current * omega_stride;
    }
}

// ============================================================
// Host-side NTT driver
// ============================================================
void ntt_gpu(Fr* d_out, const Fr* d_in, int log_n, bool inverse) {
    int n = 1 << log_n;

    if (log_n > gpu_config::NTT_MAX_LOG) {
        fprintf(stderr, "[NTT] Error: log_n=%d exceeds max %d\n",
                log_n, gpu_config::NTT_MAX_LOG);
        return;
    }

    size_t bytes = (size_t)n * sizeof(Fr);
    print_gpu_memory("NTT start");

    if (!check_gpu_memory(bytes * 2, "NTT ping-pong buffers")) {
        fprintf(stderr, "[NTT] Not enough VRAM for NTT of size 2^%d\n", log_n);
        return;
    }

    // Allocate ping-pong buffer
    Fr* d_buf;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));

    // Compute root of unity
    Fr omega;
    if (inverse) {
        omega = compute_omega_inv(log_n);
    } else {
        omega = compute_omega(log_n);
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    int half_n = n / 2;
    int bfly_blocks = (half_n + threads - 1) / threads;

    // Use shared-memory kernel for early stages where butterflies are fully
    // contained within a block (block size = 1024 elements).
    const int shared_threads = 512;
    const int shared_block_size = shared_threads * 2;
    int shared_stages = 0;
    while (shared_stages < log_n && (1 << shared_stages) < shared_block_size) {
        shared_stages++;
    }

    // Allocate twiddle buffer (reused for each stage, max size = n/2)
    Fr* d_twiddles;
    CUDA_CHECK(cudaMalloc(&d_twiddles, (size_t)half_n * sizeof(Fr)));
    Fr* h_twiddles = new Fr[half_n];

    // Cooley-Tukey (DIT) flow for both forward and inverse:
    // 1) bit-reverse input
    // 2) run stages with twiddles from omega or omega^{-1}
    bit_reverse_kernel<<<blocks, threads>>>(d_out, d_in, log_n, n);
    CUDA_CHECK(cudaGetLastError());

    if (shared_stages > 0) {
        int shared_blocks = (n + shared_block_size - 1) / shared_block_size;
        size_t shared_mem = (size_t)shared_block_size * sizeof(Fr);
        ntt_shared_kernel<<<shared_blocks, shared_threads, shared_mem>>>(
            d_out, omega, 0, shared_stages, log_n, n, inverse);
        CUDA_CHECK(cudaGetLastError());
    }

    Fr* src = d_out;
    Fr* dst = d_buf;

    for (int stage = shared_stages; stage < log_n; stage++) {
        int stride = 1 << stage;

        // Precompute and upload twiddle factors for this stage
        precompute_twiddles(h_twiddles, stride, log_n, omega);
        CUDA_CHECK(cudaMemcpy(d_twiddles, h_twiddles,
                               stride * sizeof(Fr), cudaMemcpyHostToDevice));

        ntt_butterfly_global_kernel<<<bfly_blocks, threads>>>(
            dst, src, d_twiddles, stage, log_n, n, inverse);
        CUDA_CHECK(cudaGetLastError());
        Fr* tmp = src; src = dst; dst = tmp;
    }

    if (src != d_out) {
        CUDA_CHECK(cudaMemcpy(d_out, src, bytes, cudaMemcpyDeviceToDevice));
    }

    if (inverse) {
        // Scale by 1/n = 2^{-log_n}
        // Compute (r+1)/2 in Montgomery form as the multiplicative inverse of 2.
        Fr half_mont;
        {
            uint64_t half_std[4] = {
                0xA1F0FAC9F8000001ULL,
                0x9419F4243CDCB848ULL,
                0xDC2822DB40C0AC2EULL,
                0x183227397098D014ULL
            };
            half_mont = Fr(half_std[0], half_std[1], half_std[2], half_std[3]);
            Fr r2 = Fr::r_squared();
            Fr::mont_mul_fr(half_mont.val, half_std, r2.val);
        }
        Fr scale = Fr::from_uint(1);
        for (int i = 0; i < log_n; i++) {
            scale = scale * half_mont;
        }

        ntt_scale_kernel<<<blocks, threads>>>(d_out, scale, n);
        CUDA_CHECK(cudaGetLastError());
    }

    delete[] h_twiddles;
    CUDA_CHECK(cudaFree(d_twiddles));
    CUDA_CHECK(cudaFree(d_buf));
    print_gpu_memory("NTT end");
}

void ntt_gpu_inplace(Fr* d_data, int log_n, bool inverse) {
    int n = 1 << log_n;
    size_t bytes = (size_t)n * sizeof(Fr);

    Fr* d_temp;
    CUDA_CHECK(cudaMalloc(&d_temp, bytes));
    CUDA_CHECK(cudaMemcpy(d_temp, d_data, bytes, cudaMemcpyDeviceToDevice));

    ntt_gpu(d_data, d_temp, log_n, inverse);

    CUDA_CHECK(cudaFree(d_temp));
}

} // namespace bn254
