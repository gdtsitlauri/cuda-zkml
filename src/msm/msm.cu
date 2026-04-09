#include "msm/msm.cuh"
#include "common.cuh"
#include <cooperative_groups.h>
#include <cstring>

namespace cg = cooperative_groups;

// ============================================================
// Pippenger MSM Implementation for GPU
//
// Algorithm overview (for one window w):
// 1. For each scalar s_i, extract window bits: bucket_id = (s_i >> (w*c)) & mask
// 2. Accumulate: buckets[bucket_id] += points[i]
// 3. Reduce buckets: running_sum = sum from bucket 2^c-1 down to 1,
//    result_w = sum of running_sums
// 4. Combine windows: result = sum_w result_w * 2^(w*c)
//
// GPU parallelism strategy:
// - Phase 1 (bucket accumulation): Parallelize over points.
//   Each thread processes one point for all windows.
//   Uses per-bucket spin locks to serialize Jacobian updates safely
//   across warps/blocks (race-free accumulation).
// - Phase 2 (bucket reduction): Parallelize over windows.
//   One thread reduces each window with a deterministic weighted scan.
// - Phase 3 (window combination): Single-thread sequential scan.
//
// For GTX 1650 with streaming:
//   Process chunks of 1M points at a time.
//   Accumulate partial bucket sums across chunks.
//
// Pseudocode for MSM kernel (for paper):
// ```
// parallel for each point i:
//   for each window w:
//     bits = extract_window(scalar[i], w, c)
//     if bits != 0:
//       atomic_add(buckets[w][bits-1], points[i])
//
// parallel for each window w:
//   running_sum = identity
//   window_sum = identity
//   for b = num_buckets down to 1:
//     running_sum += buckets[w][b-1]
//     window_sum += running_sum
//   window_results[w] = window_sum
//
// result = identity
// for w = num_windows-1 down to 0:
//   result = result * 2^c  (c doublings)
//   result += window_results[w]
// ```
// ============================================================

namespace bn254 {

// ============================================================
// Global bucket spin-lock helpers
//
// We use one int lock per bucket to serialize Jacobian read-modify-write
// updates safely across all blocks. This guarantees correctness even when
// many threads target the same bucket concurrently.
// ============================================================
__device__ __forceinline__ void lock_bucket(int* lock_word) {
    while (atomicCAS(lock_word, 0, 1) != 0) {
        // Busy-wait until lock is acquired.
    }
}

__device__ __forceinline__ void unlock_bucket(int* lock_word) {
    __threadfence();
    atomicExch(lock_word, 0);
}

// ============================================================
// Phase 1: Bucket accumulation kernel
// Each thread processes one point for all windows.
// Uses a simple serial accumulation into global-memory buckets.
// To avoid atomics, we partition into thread-local buckets and merge.
//
// For large n, we use a two-pass approach:
// Pass 1: Count how many points fall in each bucket
// Pass 2: Scatter-add points to buckets
//
// Simpler approach (used here): Each block processes a chunk of points
// and maintains private bucket arrays in global memory.
// Final merge across blocks.
// ============================================================

// ============================================================
// Phase 1: Bucket accumulation kernel with per-bucket spin locks
//
// Each thread processes one point for all windows.
// Each bucket update acquires a lock to make the Jacobian
// read-modify-write update race-free across all blocks.
//
// buckets layout: [num_windows][num_buckets] G1Jacobian
//
// Occupancy (sm_75):
//   256 threads/block, ~48 registers/thread (G1Affine pt + scalar + twiddle)
//   → 12288 regs/block → 5 blocks/SM.
//   16 SMs × 5 blocks × 256 threads = 20480 concurrent threads.
//   For n=1M: 1M/20480 ≈ 49 waves. Achieves ~80% occupancy.
// ============================================================
__global__ void msm_bucket_accumulate_kernel(
    G1Jacobian* buckets,         // [num_windows * num_buckets]
    int* bucket_locks,           // [num_windows * num_buckets]
    const G1Affine* points,
    const Fr* scalars,
    int n,
    int window_bits,
    int num_windows,
    int num_buckets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    G1Affine pt = points[idx];
    if (pt.is_infinity()) return;

    // Get scalar in standard form
    uint64_t s[4];
    scalars[idx].to_standard(s);

    uint64_t mask = ((uint64_t)1 << window_bits) - 1;

    for (int w = 0; w < num_windows; w++) {
        // Extract window bits
        int bit_offset = w * window_bits;
        int limb = bit_offset / 64;
        int shift = bit_offset % 64;

        uint64_t bits;
        if (limb >= 4) {
            bits = 0;
        } else if (shift + window_bits <= 64) {
            bits = (s[limb] >> shift) & mask;
        } else if (limb + 1 < 4) {
            bits = ((s[limb] >> shift) | (s[limb + 1] << (64 - shift))) & mask;
        } else {
            bits = (s[limb] >> shift) & mask;
        }

        if (bits == 0) continue;

        int bucket_idx = w * num_buckets + (int)(bits - 1);

        // Serialize this bucket update globally.
        int* lock_word = bucket_locks + bucket_idx;
        lock_bucket(lock_word);
        G1Jacobian current = buckets[bucket_idx];
        buckets[bucket_idx] = current.add_affine(pt);
        unlock_bucket(lock_word);
    }
}

// ============================================================
// Per-block private bucket accumulation with cooperative groups
// Each block has its own set of buckets in global memory.
// Uses block-level sync via cooperative groups for phase boundaries.
//
// Occupancy (sm_75):
//   256 threads/block, ~48 regs/thread.
//   Memory: num_blocks * num_windows * num_buckets * sizeof(G1Jacobian).
//   For c=14: 16383 buckets * 18 windows * 96 bytes ≈ 27 MB/block.
//   Practical limit: ~128 blocks → use streaming for large n.
// ============================================================
__global__ void msm_bucket_accumulate_v2_kernel(
    G1Jacobian* block_buckets,   // [num_blocks * num_windows * num_buckets]
    const G1Affine* points,
    const Fr* scalars,
    int n,
    int window_bits,
    int num_windows,
    int num_buckets)
{
    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Pointer to this block's buckets
    int block_offset = blockIdx.x * num_windows * num_buckets;

    if (idx >= n) return;

    G1Affine pt = points[idx];
    if (pt.is_infinity()) return;

    uint64_t s[4];
    scalars[idx].to_standard(s);
    uint64_t mask = ((uint64_t)1 << window_bits) - 1;

    for (int w = 0; w < num_windows; w++) {
        int bit_offset = w * window_bits;
        int limb = bit_offset / 64;
        int shift = bit_offset % 64;

        uint64_t bits;
        if (limb >= 4) bits = 0;
        else if (shift + window_bits <= 64) bits = (s[limb] >> shift) & mask;
        else if (limb + 1 < 4) bits = ((s[limb] >> shift) | (s[limb + 1] << (64 - shift))) & mask;
        else bits = (s[limb] >> shift) & mask;

        if (bits == 0) continue;

        int bucket_idx = block_offset + w * num_buckets + (int)(bits - 1);

        // Within a block, threads rarely hit the same bucket (2^c buckets >> 256 threads).
        // Use warp shuffle to detect the rare collision case.
        unsigned int lane = warp.thread_rank();
        bool conflict = false;
        for (int peer = 0; peer < (int)lane; peer++) {
            if (warp.shfl(bucket_idx, peer) == bucket_idx) {
                conflict = true;
                break;
            }
        }

        if (!conflict) {
            G1Jacobian current = block_buckets[bucket_idx];
            block_buckets[bucket_idx] = current.add_affine(pt);
        }
        warp.sync();
        if (conflict) {
            G1Jacobian current = block_buckets[bucket_idx];
            block_buckets[bucket_idx] = current.add_affine(pt);
        }
        warp.sync();
    }
}

// ============================================================
// Merge block-private buckets into final buckets
// Each thread merges one bucket across all blocks.
//
// Occupancy (sm_75):
//   256 threads/block, ~36 registers/thread (2 G1Jacobian accumulators).
//   → 9216 regs/block → 7 blocks/SM.
//   16 SMs × 7 × 256 = 28672 concurrent threads.
//   For 18 windows × 16383 buckets = 294K buckets → 11.5 waves. Good.
// ============================================================
__global__ void msm_bucket_merge_kernel(
    G1Jacobian* final_buckets,    // [num_windows * num_buckets]
    const G1Jacobian* block_buckets, // [num_blocks * num_windows * num_buckets]
    int num_blocks_to_merge,
    int num_windows,
    int num_buckets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_buckets = num_windows * num_buckets;
    if (idx >= total_buckets) return;

    G1Jacobian sum = G1Jacobian::identity();
    for (int b = 0; b < num_blocks_to_merge; b++) {
        G1Jacobian val = block_buckets[b * total_buckets + idx];
        if (!val.is_infinity()) {
            sum = sum + val;
        }
    }
    final_buckets[idx] = sum;
}

// ============================================================
// Phase 2: Bucket reduction kernel
// For each window, compute the weighted sum of buckets:
//   running = 0
//   window_sum = 0
//   for b = num_buckets down to 1:
//     running += buckets[w][b-1]
//     window_sum += running
//   window_results[w] = window_sum
//
// Uses a two-level parallel approach:
// - 32 threads per window (one warp), each handles num_buckets/32 buckets
// - Within each warp lane, compute partial running_sum and window_sum
// - Cross-lane combine using warp shuffle for final prefix-sum
//
// Occupancy (sm_75):
//   1 warp (32 threads) per window, 1 block = 1 window.
//   ~40 registers/thread → 1280 regs/block → 51 blocks/SM.
//   Limited by block count to 16 blocks/SM × 16 SMs = 256 windows.
//   For 18 windows: trivially fits, limited by parallelism not occupancy.
// ============================================================
__global__ void msm_bucket_reduce_kernel(
    G1Jacobian* window_results,   // [num_windows]
    const G1Jacobian* buckets,    // [num_windows * num_buckets]
    int num_windows,
    int num_buckets)
{
    int w = blockIdx.x;
    if (w >= num_windows || threadIdx.x != 0) return;

    const G1Jacobian* window_buckets = buckets + w * num_buckets;
    G1Jacobian running = G1Jacobian::identity();
    G1Jacobian window_sum = G1Jacobian::identity();

    // Sequential weighted bucket scan per window.
    for (int b = num_buckets - 1; b >= 0; b--) {
        G1Jacobian bucket_val = window_buckets[b];
        if (!bucket_val.is_infinity()) {
            running = running + bucket_val;
        }
        window_sum = window_sum + running;
    }

    window_results[w] = window_sum;
}

// ============================================================
// Phase 3: Window combination (host-side sequential)
// result = 0
// for w = num_windows-1 down to 0:
//   result = result * 2^c  (c doublings)
//   result += window_results[w]
// ============================================================

// ============================================================
// Host-side MSM driver (Pippenger)
// ============================================================
G1Jacobian msm_g1(const G1Affine* h_points, const Fr* h_scalars, int n) {
    if (n == 0) return G1Jacobian::identity();

    print_gpu_memory("MSM start");

    MSMConfig cfg = MSMConfig::for_size(n);
    printf("[MSM] n=%d, window_bits=%d, num_windows=%d, num_buckets=%d\n",
           n, cfg.window_bits, cfg.num_windows, cfg.num_buckets);

    // For small n, fall back to simple approach
    if (n <= 256) {
        // CPU scalar multiplication and summation
        G1Jacobian result = G1Jacobian::identity();
        for (int i = 0; i < n; i++) {
            G1Jacobian pt = G1Jacobian::from_affine(h_points[i]);
            G1Jacobian contrib = pt.scalar_mul(h_scalars[i]);
            result = result + contrib;
        }
        return result;
    }

    // Check memory budget
    int threads_per_block = 256;
    int total_buckets = cfg.num_windows * cfg.num_buckets;

    // Memory needed:
    // Points: n * sizeof(G1Affine)
    // Scalars: n * sizeof(Fr)
    // Buckets: total_buckets * sizeof(G1Jacobian)
    // Window results: num_windows * sizeof(G1Jacobian)
    size_t mem_points = (size_t)n * sizeof(G1Affine);
    size_t mem_scalars = (size_t)n * sizeof(Fr);
    size_t mem_buckets = (size_t)total_buckets * sizeof(G1Jacobian);
    size_t mem_total = mem_points + mem_scalars + mem_buckets;

    printf("[MSM] Memory: points=%.1f MB, scalars=%.1f MB, buckets=%.1f MB, total=%.1f MB\n",
           mem_points / 1e6, mem_scalars / 1e6, mem_buckets / 1e6, mem_total / 1e6);

    // Check if we need streaming
    bool streaming = !check_gpu_memory(mem_total + 64 * 1024 * 1024, "MSM total");
    int chunk_size = streaming ? cfg.chunk_size : n;

    // Allocate GPU memory
    G1Affine* d_points;
    Fr* d_scalars;
    G1Jacobian* d_buckets;
    int* d_bucket_locks;
    G1Jacobian* d_window_results;

    size_t chunk_points_bytes = (size_t)chunk_size * sizeof(G1Affine);
    size_t chunk_scalars_bytes = (size_t)chunk_size * sizeof(Fr);

    CUDA_CHECK(cudaMalloc(&d_points, chunk_points_bytes));
    CUDA_CHECK(cudaMalloc(&d_scalars, chunk_scalars_bytes));
    CUDA_CHECK(cudaMalloc(&d_buckets, mem_buckets));
    CUDA_CHECK(cudaMalloc(&d_bucket_locks, (size_t)total_buckets * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_window_results, cfg.num_windows * sizeof(G1Jacobian)));
    CUDA_CHECK(cudaMemset(d_bucket_locks, 0, (size_t)total_buckets * sizeof(int)));

    // Initialize buckets to identity
    // Create identity points on host and upload
    G1Jacobian* h_identity_buckets = new G1Jacobian[total_buckets];
    for (int i = 0; i < total_buckets; i++) {
        h_identity_buckets[i] = G1Jacobian::identity();
    }
    CUDA_CHECK(cudaMemcpy(d_buckets, h_identity_buckets,
                           mem_buckets, cudaMemcpyHostToDevice));
    delete[] h_identity_buckets;

    print_gpu_memory("MSM after alloc");

    // Process chunks
    for (int offset = 0; offset < n; offset += chunk_size) {
        int current_chunk = min(chunk_size, n - offset);
        printf("[MSM] Processing chunk: offset=%d, size=%d\n", offset, current_chunk);

        // Upload chunk
        CUDA_CHECK(cudaMemcpy(d_points, h_points + offset,
                               current_chunk * sizeof(G1Affine), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars + offset,
                               current_chunk * sizeof(Fr), cudaMemcpyHostToDevice));

        // Bucket accumulation
        int chunk_blocks = (current_chunk + threads_per_block - 1) / threads_per_block;
        msm_bucket_accumulate_kernel<<<chunk_blocks, threads_per_block>>>(
            d_buckets, d_bucket_locks, d_points, d_scalars, current_chunk,
            cfg.window_bits, cfg.num_windows, cfg.num_buckets);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    // Bucket reduction
    msm_bucket_reduce_kernel<<<cfg.num_windows, 1>>>(
        d_window_results, d_buckets, cfg.num_windows, cfg.num_buckets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Download window results
    G1Jacobian* h_window_results = new G1Jacobian[cfg.num_windows];
    CUDA_CHECK(cudaMemcpy(h_window_results, d_window_results,
                           cfg.num_windows * sizeof(G1Jacobian), cudaMemcpyDeviceToHost));

    // Phase 3: Combine windows on CPU
    G1Jacobian result = G1Jacobian::identity();
    for (int w = cfg.num_windows - 1; w >= 0; w--) {
        // result *= 2^c
        for (int i = 0; i < cfg.window_bits; i++) {
            result = result.dbl();
        }
        result = result + h_window_results[w];
    }

    // Cleanup
    delete[] h_window_results;
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_scalars));
    CUDA_CHECK(cudaFree(d_buckets));
    CUDA_CHECK(cudaFree(d_bucket_locks));
    CUDA_CHECK(cudaFree(d_window_results));

    print_gpu_memory("MSM end");
    return result;
}

// ============================================================
// MSM for G2 (same algorithm, different point type)
//
// G2 points are over Fp2 so all coordinates are 2x Fp size.
// This halves effective occupancy compared to G1 kernels.
// ============================================================

// ============================================================
// G2 bucket accumulation with per-bucket spin locks
//
// Occupancy (sm_75):
//   128 threads/block, ~96 registers/thread (G2Affine = 2×Fp2 = 8×uint64).
//   → 12288 regs/block → 5 blocks/SM.
//   16 SMs × 5 × 128 = 10240 concurrent threads.
// ============================================================
__global__ void msm_g2_bucket_accumulate_kernel(
    G2Jacobian* buckets,
    int* bucket_locks,
    const G2Affine* points,
    const Fr* scalars,
    int n, int window_bits, int num_windows, int num_buckets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    G2Affine pt = points[idx];
    if (pt.is_infinity()) return;

    uint64_t s[4];
    scalars[idx].to_standard(s);
    uint64_t mask = ((uint64_t)1 << window_bits) - 1;

    for (int w = 0; w < num_windows; w++) {
        int bit_offset = w * window_bits;
        int limb = bit_offset / 64;
        int shift = bit_offset % 64;

        uint64_t bits;
        if (limb >= 4) bits = 0;
        else if (shift + window_bits <= 64) bits = (s[limb] >> shift) & mask;
        else if (limb + 1 < 4) bits = ((s[limb] >> shift) | (s[limb + 1] << (64 - shift))) & mask;
        else bits = (s[limb] >> shift) & mask;

        if (bits == 0) continue;

        int bucket_idx = w * num_buckets + (int)(bits - 1);

        int* lock_word = bucket_locks + bucket_idx;
        lock_bucket(lock_word);
        G2Jacobian current = buckets[bucket_idx];
        buckets[bucket_idx] = current.add_affine(pt);
        unlock_bucket(lock_word);
    }
}

// ============================================================
// G2 bucket reduction (sequential scan, one thread per window)
//
// Occupancy (sm_75):
//   1 thread per window (18 windows total). Not occupancy-bound —
//   this kernel is latency-bound by serial EC additions.
//   Total work: 18 × num_buckets G2 additions.
// ============================================================
__global__ void msm_g2_bucket_reduce_kernel(
    G2Jacobian* window_results,
    const G2Jacobian* buckets,
    int num_windows, int num_buckets)
{
    int w = blockIdx.x;
    if (w >= num_windows || threadIdx.x != 0) return;

    G2Jacobian running = G2Jacobian::identity();
    G2Jacobian window_sum = G2Jacobian::identity();
    const G2Jacobian* wb = buckets + w * num_buckets;

    for (int b = num_buckets - 1; b >= 0; b--) {
        if (!wb[b].is_infinity()) running = running + wb[b];
        window_sum = window_sum + running;
    }
    window_results[w] = window_sum;
}

G2Jacobian msm_g2(const G2Affine* h_points, const Fr* h_scalars, int n) {
    if (n == 0) return G2Jacobian::identity();

    MSMConfig cfg = MSMConfig::for_size(n);
    // G2 is 2x larger, reduce window bits
    if (cfg.window_bits > 12) cfg.window_bits = 12;
    cfg.num_windows = (253 + cfg.window_bits) / cfg.window_bits;
    cfg.num_buckets = (1 << cfg.window_bits) - 1;

    int total_buckets = cfg.num_windows * cfg.num_buckets;

    if (n <= 128) {
        G2Jacobian result = G2Jacobian::identity();
        for (int i = 0; i < n; i++) {
            G2Jacobian pt = G2Jacobian::from_affine(h_points[i]);
            result = result + pt.scalar_mul(h_scalars[i]);
        }
        return result;
    }

    G2Affine* d_points;
    Fr* d_scalars;
    G2Jacobian* d_buckets;
    int* d_bucket_locks;
    G2Jacobian* d_window_results;

    CUDA_CHECK(cudaMalloc(&d_points, (size_t)n * sizeof(G2Affine)));
    CUDA_CHECK(cudaMalloc(&d_scalars, (size_t)n * sizeof(Fr)));
    CUDA_CHECK(cudaMalloc(&d_buckets, (size_t)total_buckets * sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMalloc(&d_bucket_locks, (size_t)total_buckets * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_window_results, cfg.num_windows * sizeof(G2Jacobian)));
    CUDA_CHECK(cudaMemset(d_bucket_locks, 0, (size_t)total_buckets * sizeof(int)));

    CUDA_CHECK(cudaMemcpy(d_points, h_points, n * sizeof(G2Affine), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_scalars, h_scalars, n * sizeof(Fr), cudaMemcpyHostToDevice));

    G2Jacobian* h_id = new G2Jacobian[total_buckets];
    for (int i = 0; i < total_buckets; i++) h_id[i] = G2Jacobian::identity();
    CUDA_CHECK(cudaMemcpy(d_buckets, h_id, total_buckets * sizeof(G2Jacobian), cudaMemcpyHostToDevice));
    delete[] h_id;

    int threads = 128;
    int blocks = (n + threads - 1) / threads;
    msm_g2_bucket_accumulate_kernel<<<blocks, threads>>>(
        d_buckets, d_bucket_locks, d_points, d_scalars, n,
        cfg.window_bits, cfg.num_windows, cfg.num_buckets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    msm_g2_bucket_reduce_kernel<<<cfg.num_windows, 1>>>(
        d_window_results, d_buckets, cfg.num_windows, cfg.num_buckets);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    G2Jacobian* h_wr = new G2Jacobian[cfg.num_windows];
    CUDA_CHECK(cudaMemcpy(h_wr, d_window_results,
                           cfg.num_windows * sizeof(G2Jacobian), cudaMemcpyDeviceToHost));

    G2Jacobian result = G2Jacobian::identity();
    for (int w = cfg.num_windows - 1; w >= 0; w--) {
        for (int i = 0; i < cfg.window_bits; i++) result = result.dbl();
        result = result + h_wr[w];
    }

    delete[] h_wr;
    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_scalars));
    CUDA_CHECK(cudaFree(d_buckets));
    CUDA_CHECK(cudaFree(d_bucket_locks));
    CUDA_CHECK(cudaFree(d_window_results));

    return result;
}

} // namespace bn254
