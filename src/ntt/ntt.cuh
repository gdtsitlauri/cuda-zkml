#pragma once

#include "field/fp.cuh"

// ============================================================
// Number Theoretic Transform (NTT) over BN254 scalar field Fr
//
// NTT is the finite field analogue of FFT. Used for:
// - Polynomial multiplication in O(n log n)
// - Groth16 proving (evaluating polynomials at roots of unity)
// - Witness generation
//
// BN254 scalar field r has 2-adicity of 28:
//   r - 1 = 2^28 * k for odd k
//   Max NTT size = 2^28 (but we cap at 2^24 for 4GB VRAM)
//
// Implementation: Gentleman-Sande (decimation-in-frequency) butterfly
// with bit-reversal permutation.
//
// Memory layout: out-of-place with ping-pong buffers
// Max allocation: 2^24 * 32 bytes = 512 MB per buffer (fits in VRAM)
//
// Occupancy (sm_75):
//   Each NTT butterfly: 2 Fr muls + 2 Fr adds = ~40 registers
//   With shared memory optimization for intra-block stages:
//   1024 threads/block, 48KB shared memory for 1536 Fr elements
//   Each thread handles one butterfly → 512 butterflies per block
// ============================================================

namespace bn254 {

// Primitive root of unity for Fr
// Generator of the multiplicative group of Fr
// g = 5 (a primitive root mod r)
// omega_2^28 = g^((r-1)/2^28) = root of unity of order 2^28
//
// Precomputed: omega = 5^((r-1)/2^28) mod r in Montgomery form
// For 2-adicity 28:
//   r - 1 = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000000
//   (r-1)/2^28 = 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f

struct NTTConfig {
    int log_n;          // log2(n)
    int n;              // transform size
    bool inverse;       // true for iNTT
};

// Host function declarations
void ntt_gpu(Fr* d_out, const Fr* d_in, int log_n, bool inverse);
void ntt_gpu_inplace(Fr* d_data, int log_n, bool inverse);

// Compute omega (root of unity) for given order 2^k
__host__ __device__ Fr compute_omega(int log_n);
__host__ __device__ Fr compute_omega_inv(int log_n);

} // namespace bn254
