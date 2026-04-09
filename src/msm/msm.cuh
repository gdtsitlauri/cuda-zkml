#pragma once

#include "curve/g1.cuh"
#include "curve/g2.cuh"

// ============================================================
// Multi-Scalar Multiplication (MSM)
//
// Computes: result = sum_{i=0}^{n-1} scalars[i] * points[i]
//
// This is THE performance-critical operation in ZK proving.
// A Groth16 proof requires 3-4 MSMs over G1 and 1-2 over G2.
//
// Algorithm: Pippenger's bucket method
// 1. Partition scalar bits into windows of size c
// 2. For each window, sort points into 2^c buckets by window bits
// 3. Sum each bucket using parallel prefix scan
// 4. Combine windows: result = sum_w bucket_sums[w] * 2^(w*c)
//
// Optimal window size for n points: c ≈ log2(n)
// For n = 2^16: c = 16, buckets = 65536
// For n = 2^20: c = 20, buckets = 1M → too many, use c = 15
//
// Memory for GTX 1650 (3.5GB VRAM):
//   Points: n * 96 bytes (G1Jacobian)
//   Scalars: n * 32 bytes
//   Buckets: 2^c * 96 bytes
//   For n = 1M, c = 15: 1M * 128 + 32K * 96 = 131 MB. Fine.
//   For n > 1M: use streaming MSM (process in chunks)
//
// Occupancy (sm_75):
//   Bucket accumulation: 256 threads/block, ~40 regs/thread
//   → 10240 regs/block → 6 blocks/SM. 16 SMs → 96 blocks.
//   Bucket reduction: 128 threads/block (register-heavy due to adds)
// ============================================================

namespace bn254 {

// MSM configuration
struct MSMConfig {
    int window_bits;   // c: bits per window
    int num_windows;   // ceil(254 / c)
    int num_buckets;   // 2^c - 1 (bucket 0 = identity, skip)
    int chunk_size;    // for streaming MSM

    static MSMConfig for_size(int n) {
        MSMConfig cfg;
        // Optimal window size: approximately log2(n)
        if (n <= (1 << 10)) cfg.window_bits = 10;
        else if (n <= (1 << 14)) cfg.window_bits = 12;
        else if (n <= (1 << 18)) cfg.window_bits = 14;
        else if (n <= (1 << 22)) cfg.window_bits = 15;
        else cfg.window_bits = 16;

        cfg.num_windows = (253 + cfg.window_bits) / cfg.window_bits;
        cfg.num_buckets = (1 << cfg.window_bits) - 1;

        // Streaming chunk size for 4GB VRAM
        // Each point = 96 bytes, each scalar = 32 bytes = 128 bytes/element
        // Budget 2GB for points+scalars: 2GB / 128 = ~16M elements
        cfg.chunk_size = gpu_config::MSM_CHUNK_SIZE; // 1M default
        return cfg;
    }
};

// Host-side MSM function declarations
G1Jacobian msm_g1(const G1Affine* h_points, const Fr* h_scalars, int n);
G2Jacobian msm_g2(const G2Affine* h_points, const Fr* h_scalars, int n);

} // namespace bn254
