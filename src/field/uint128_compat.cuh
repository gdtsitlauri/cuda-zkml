#pragma once

// ============================================================
// Portable 128-bit unsigned integer for host+device code
//
// On device (__CUDA_ARCH__ defined): use native unsigned __int128
// On host with GCC/Clang: use native unsigned __int128
// On host with MSVC: use _umul128 / _addcarry_u64 intrinsics
//
// This header provides macros and inline functions so that
// __host__ __device__ functions can use 128-bit arithmetic
// portably across MSVC and GCC/Clang host compilers.
// ============================================================

#include <cstdint>

// ============================================================
// Detect whether we have native __int128 support
// - Always available in __device__ code (NVCC supports it)
// - Available on host with GCC/Clang
// - NOT available on MSVC host
// ============================================================

#if defined(__CUDA_ARCH__) || !defined(_MSC_VER)
// Device code or non-MSVC host: native __int128 is available
#define UINT128_NATIVE 1
#else
// MSVC host code: need intrinsics
#define UINT128_NATIVE 0
#include <intrin.h>
#pragma intrinsic(_umul128, _addcarry_u64, _subborrow_u64)
#endif

// ============================================================
// Portable 128-bit operations as inline functions
// These are used in __host__ __device__ functions where
// we need to work on both device (always has __int128) and
// MSVC host (never has __int128).
//
// Strategy: use #ifdef __CUDA_ARCH__ inside each function.
// When compiled for device: use __int128.
// When compiled for host:
//   - GCC/Clang: use __int128
//   - MSVC: use intrinsics
// ============================================================

// Multiply two 64-bit values, return low 64 bits, store high in *hi
__host__ __device__ __forceinline__
uint64_t mul64_full(uint64_t a, uint64_t b, uint64_t* hi) {
#ifdef __CUDA_ARCH__
    *hi = __umul64hi(a, b);
    return a * b;
#elif defined(_MSC_VER)
    return _umul128(a, b, hi);
#else
    unsigned __int128 r = (unsigned __int128)a * b;
    *hi = (uint64_t)(r >> 64);
    return (uint64_t)r;
#endif
}

// Add with carry: result = a + b + carry_in, returns carry_out
__host__ __device__ __forceinline__
uint64_t adc64(uint64_t a, uint64_t b, uint64_t carry_in, uint64_t* result) {
    uint64_t s = a + b;
    uint64_t c1 = (s < a) ? 1ULL : 0ULL;
    uint64_t s2 = s + carry_in;
    uint64_t c2 = (s2 < s) ? 1ULL : 0ULL;
    *result = s2;
    return c1 + c2;
}

// Subtract with borrow: result = a - b - borrow_in, returns borrow_out (0 or 1)
__host__ __device__ __forceinline__
uint64_t sbb64(uint64_t a, uint64_t b, uint64_t borrow_in, uint64_t* result) {
    uint64_t d = a - b;
    uint64_t b1 = (a < b) ? 1ULL : 0ULL;
    uint64_t d2 = d - borrow_in;
    uint64_t b2 = (d < borrow_in) ? 1ULL : 0ULL;
    *result = d2;
    return b1 + b2;
}

// Multiply-accumulate: acc += a * b, where acc is a (lo, hi) pair
// Returns new (lo, hi) accumulator
__host__ __device__ __forceinline__
void mac64(uint64_t a, uint64_t b, uint64_t* acc_lo, uint64_t* acc_hi) {
#ifdef __CUDA_ARCH__
    uint64_t prod_hi;
    uint64_t prod_lo = mul64_full(a, b, &prod_hi);
    uint64_t new_lo;
    uint64_t c = adc64(*acc_lo, prod_lo, 0, &new_lo);
    uint64_t new_hi;
    adc64(*acc_hi, prod_hi, c, &new_hi);
    *acc_lo = new_lo;
    *acc_hi = new_hi;
#elif defined(_MSC_VER)
    uint64_t prod_hi;
    uint64_t prod_lo = _umul128(a, b, &prod_hi);
    unsigned char c = _addcarry_u64(0, *acc_lo, prod_lo, acc_lo);
    _addcarry_u64(c, *acc_hi, prod_hi, acc_hi);
#else
    unsigned __int128 prod = (unsigned __int128)a * b;
    unsigned __int128 acc = ((unsigned __int128)(*acc_hi) << 64) | *acc_lo;
    acc += prod;
    *acc_lo = (uint64_t)acc;
    *acc_hi = (uint64_t)(acc >> 64);
#endif
}
