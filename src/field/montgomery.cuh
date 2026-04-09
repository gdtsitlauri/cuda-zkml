#pragma once

#include "common.cuh"
#include "field/uint128_compat.cuh"

// ============================================================
// Montgomery Multiplication for 256-bit modular arithmetic
//
// BN254 prime p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
// In hex: 0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47
//
// Montgomery form: aR mod p, where R = 2^256
// Montgomery multiplication: MonMul(aR, bR) = abR mod p
//
// Key constants:
//   R = 2^256 mod p
//   R^2 = (2^256)^2 mod p
//   p' = -p^{-1} mod 2^64 (for CIOS reduction)
//
// Occupancy for sm_75 (GTX 1650, Turing):
//   Each thread uses 4x uint64 = 8 registers for each operand,
//   plus ~20 scratch registers for CIOS inner loop.
//   Total: ~32 registers per thread.
//   With 256 threads/block: 32 * 256 = 8192 registers/block.
//   sm_75 has 65536 regs/SM, 16 SMs.
//   Max blocks/SM = min(65536/8192, 16) = 8 blocks/SM.
//   Achieved occupancy: 8 * 256 / 1024 = 200% of single-wave → good.
// ============================================================

namespace bn254 {

// BN254 base field prime p
__device__ __constant__ const uint64_t P_LIMBS[4] = {
    0x3C208C16D87CFD47ULL,  // p[0] - least significant
    0x97816A916871CA8DULL,  // p[1]
    0xB85045B68181585DULL,  // p[2]
    0x30644E72E131A029ULL   // p[3] - most significant
};

// -p^{-1} mod 2^64  (Montgomery constant for CIOS)
// Verified: p * 0x87D20782E4866389 ≡ -1 (mod 2^64)
__device__ __constant__ const uint64_t P_INV = 0x87D20782E4866389ULL;

// R mod p = 2^256 mod p
// = 0x0e0a77c19a07df2f666ea36f7879462c0a78eb28f5c70b3dd35d438dc58f0d9d
__device__ __constant__ const uint64_t R_MOD_P[4] = {
    0xD35D438DC58F0D9DULL,
    0x0A78EB28F5C70B3DULL,
    0x666EA36F7879462CULL,
    0x0E0A77C19A07DF2FULL
};

// R^2 mod p = (2^256)^2 mod p
// = 0x06d89f71cab8351f47ab1eff0a417ff6b5e71911d44501fbf32cfc5b538afa89
__device__ __constant__ const uint64_t R2_MOD_P[4] = {
    0xF32CFC5B538AFA89ULL,
    0xB5E71911D44501FBULL,
    0x47AB1EFF0A417FF6ULL,
    0x06D89F71CAB8351FULL
};

// ============================================================
// Host-side constants (duplicated because __constant__ is device-only)
// ============================================================
static const uint64_t HOST_P[4] = {
    0x3C208C16D87CFD47ULL, 0x97816A916871CA8DULL,
    0xB85045B68181585DULL, 0x30644E72E131A029ULL
};
static const uint64_t HOST_P_INV = 0x87D20782E4866389ULL;
static const uint64_t HOST_R2[4] = {
    0xF32CFC5B538AFA89ULL, 0xB5E71911D44501FBULL,
    0x47AB1EFF0A417FF6ULL, 0x06D89F71CAB8351FULL
};

// ============================================================
// PTX-accelerated 256-bit addition with carry for sm_75
// Uses add.cc / addc.cc carry chain instructions.
// Returns carry out (0 or 1).
// ============================================================
__device__ __forceinline__
uint64_t add_256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t carry;
    asm(
        "add.cc.u64   %0, %5, %9;\n\t"
        "addc.cc.u64  %1, %6, %10;\n\t"
        "addc.cc.u64  %2, %7, %11;\n\t"
        "addc.cc.u64  %3, %8, %12;\n\t"
        "addc.u64     %4, 0, 0;\n\t"
        : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3]), "=l"(carry)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
    );
    return carry;
}

// ============================================================
// PTX-accelerated 256-bit subtraction with borrow for sm_75
// Uses sub.cc / subc.cc borrow chain instructions.
// Returns borrow out (0 or 1).
// ============================================================
__device__ __forceinline__
uint64_t sub_256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    uint64_t borrow;
    asm(
        "sub.cc.u64   %0, %5, %9;\n\t"
        "subc.cc.u64  %1, %6, %10;\n\t"
        "subc.cc.u64  %2, %7, %11;\n\t"
        "subc.cc.u64  %3, %8, %12;\n\t"
        "subc.u64     %4, 0, 0;\n\t"    // borrow is 0xFFFF... if borrow, 0 otherwise
        : "=l"(r[0]), "=l"(r[1]), "=l"(r[2]), "=l"(r[3]), "=l"(borrow)
        : "l"(a[0]), "l"(a[1]), "l"(a[2]), "l"(a[3]),
          "l"(b[0]), "l"(b[1]), "l"(b[2]), "l"(b[3])
    );
    return borrow & 1; // normalize to 0 or 1
}

// ============================================================
// Host-side 256-bit add/sub
// On GCC/Clang: use unsigned __int128 for carry propagation.
// On MSVC: use _addcarry_u64 / _subborrow_u64 intrinsics.
// ============================================================
#if defined(_MSC_VER)
#include <intrin.h>

__host__ inline
uint64_t host_add_256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    unsigned char carry = 0;
    carry = _addcarry_u64(carry, a[0], b[0], &r[0]);
    carry = _addcarry_u64(carry, a[1], b[1], &r[1]);
    carry = _addcarry_u64(carry, a[2], b[2], &r[2]);
    carry = _addcarry_u64(carry, a[3], b[3], &r[3]);
    return (uint64_t)carry;
}

__host__ inline
uint64_t host_sub_256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    unsigned char borrow = 0;
    borrow = _subborrow_u64(borrow, a[0], b[0], &r[0]);
    borrow = _subborrow_u64(borrow, a[1], b[1], &r[1]);
    borrow = _subborrow_u64(borrow, a[2], b[2], &r[2]);
    borrow = _subborrow_u64(borrow, a[3], b[3], &r[3]);
    return (uint64_t)borrow;
}

#else // GCC / Clang

__host__ inline
uint64_t host_add_256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    unsigned __int128 acc = (unsigned __int128)a[0] + b[0];
    r[0] = (uint64_t)acc;
    acc = (unsigned __int128)a[1] + b[1] + (uint64_t)(acc >> 64);
    r[1] = (uint64_t)acc;
    acc = (unsigned __int128)a[2] + b[2] + (uint64_t)(acc >> 64);
    r[2] = (uint64_t)acc;
    acc = (unsigned __int128)a[3] + b[3] + (uint64_t)(acc >> 64);
    r[3] = (uint64_t)acc;
    return (uint64_t)(acc >> 64);
}

__host__ inline
uint64_t host_sub_256(uint64_t* r, const uint64_t* a, const uint64_t* b) {
    unsigned __int128 d;
    uint64_t borrow = 0;
    d = (unsigned __int128)a[0] - b[0];               r[0] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (unsigned __int128)a[1] - b[1] - borrow;      r[1] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (unsigned __int128)a[2] - b[2] - borrow;      r[2] = (uint64_t)d; borrow = (d >> 127) & 1;
    d = (unsigned __int128)a[3] - b[3] - borrow;      r[3] = (uint64_t)d; borrow = (d >> 127) & 1;
    return borrow;
}

#endif

// ============================================================
// CIOS Montgomery multiplication (device)
//
// Computes result = a * b * R^{-1} mod p
// where a, b are in Montgomery form.
//
// Algorithm: Coarsely Integrated Operand Scanning
// For each limb i of a:
//   1. Accumulate partial product a[i] * b into T
//   2. Compute reduction factor m = T[0] * (-p^{-1}) mod 2^64
//   3. Add m * p to T, shift right by 64 bits
// After 4 iterations, T holds the result (with possible final subtraction).
//
// Cost: 4 * (4 mul + 4 madd) = 32 uint64 multiplications
// ============================================================
__device__ __forceinline__
void montgomery_mul_cios(uint64_t* result, const uint64_t* a, const uint64_t* b) {
    const uint64_t* p = P_LIMBS;
    const uint64_t inv = P_INV;

    uint64_t T[5] = {0, 0, 0, 0, 0};

    #pragma unroll
    for (int i = 0; i < 4; i++) {
        // Step 1: T = T + a[i] * b
        uint64_t carry_lo = 0;
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t hi;
            uint64_t lo = mul64_full(a[i], b[j], &hi);

            uint64_t t;
            uint64_t c1 = adc64(T[j], lo, 0, &t);
            uint64_t c2 = adc64(t, carry_lo, 0, &T[j]);
            carry_lo = hi + c1 + c2;
        }
        adc64(T[4], carry_lo, 0, &T[4]);

        // Step 2: m = T[0] * inv mod 2^64
        uint64_t m = T[0] * inv;

        // Step 3: T = (T + m * p) >> 64
        carry_lo = 0;
        {
            uint64_t hi;
            uint64_t lo = mul64_full(m, p[0], &hi);
            uint64_t sum_lo;
            uint64_t sum_carry = adc64(T[0], lo, 0, &sum_lo);
            carry_lo = hi + sum_carry;
        }

        #pragma unroll
        for (int j = 1; j < 4; j++) {
            uint64_t hi;
            uint64_t lo = mul64_full(m, p[j], &hi);

            uint64_t t1;
            uint64_t c1 = adc64(T[j], lo, 0, &t1);
            uint64_t t2;
            uint64_t c2 = adc64(t1, carry_lo, 0, &t2);
            T[j - 1] = t2;
            carry_lo = hi + c1 + c2;
        }
        uint64_t t4 = T[4];
        T[3] = carry_lo + t4;
        T[4] = (T[3] < carry_lo || T[3] < t4) ? 1 : 0;
    }

    result[0] = T[0]; result[1] = T[1]; result[2] = T[2]; result[3] = T[3];

    // Final reduction: if result >= p, subtract p
    uint64_t tmp[4];
    uint64_t borrow = sub_256(tmp, result, p);
    if (borrow == 0) {
        result[0] = tmp[0]; result[1] = tmp[1];
        result[2] = tmp[2]; result[3] = tmp[3];
    }
}

// ============================================================
// Device helpers: to/from Montgomery form
// ============================================================
__device__ __forceinline__
void to_montgomery(uint64_t* result, const uint64_t* a) {
    montgomery_mul_cios(result, a, R2_MOD_P);
}

__device__ __forceinline__
void from_montgomery(uint64_t* result, const uint64_t* a) {
    uint64_t one[4] = {1, 0, 0, 0};
    montgomery_mul_cios(result, a, one);
}

// ============================================================
// Host-side CIOS Montgomery multiplication
// Uses __int128 on GCC/Clang, _umul128 + _addcarry on MSVC.
// ============================================================
#if defined(_MSC_VER)

__host__ inline
void host_montgomery_mul(uint64_t* result, const uint64_t* a, const uint64_t* b) {
    const uint64_t* p = HOST_P;
    const uint64_t inv = HOST_P_INV;

    uint64_t T[5] = {0, 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        // Step 1: T = T + a[i] * b
        uint64_t carry_lo = 0;
        for (int j = 0; j < 4; j++) {
            uint64_t hi;
            uint64_t lo = mul64_full(a[i], b[j], &hi);
            uint64_t t;
            uint64_t c1 = adc64(T[j], lo, 0, &t);
            uint64_t c2 = adc64(t, carry_lo, 0, &T[j]);
            carry_lo = hi + c1 + c2;
        }
        adc64(T[4], carry_lo, 0, &T[4]);

        // Step 2: m = T[0] * inv mod 2^64
        uint64_t m = T[0] * inv;

        // Step 3: T = (T + m * p) >> 64
        carry_lo = 0;
        {
            uint64_t hi;
            uint64_t lo = mul64_full(m, p[0], &hi);
            uint64_t sum_lo;
            uint64_t sum_carry = adc64(T[0], lo, 0, &sum_lo);
            carry_lo = hi + sum_carry;
        }
        for (int j = 1; j < 4; j++) {
            uint64_t hi;
            uint64_t lo = mul64_full(m, p[j], &hi);
            uint64_t t1;
            uint64_t c1 = adc64(T[j], lo, 0, &t1);
            uint64_t c2 = adc64(t1, carry_lo, 0, &T[j - 1]);
            carry_lo = hi + c1 + c2;
        }
        uint64_t t4 = T[4];
        T[3] = carry_lo + t4;
        T[4] = (T[3] < carry_lo || T[3] < t4) ? 1 : 0;
    }

    result[0] = T[0]; result[1] = T[1]; result[2] = T[2]; result[3] = T[3];

    uint64_t tmp[4];
    uint64_t borrow = host_sub_256(tmp, result, p);
    if (borrow == 0) {
        result[0] = tmp[0]; result[1] = tmp[1];
        result[2] = tmp[2]; result[3] = tmp[3];
    }
}

#else // GCC / Clang

__host__ inline
void host_montgomery_mul(uint64_t* result, const uint64_t* a, const uint64_t* b) {
    const uint64_t* p = HOST_P;
    const uint64_t inv = HOST_P_INV;

    uint64_t T[5] = {0, 0, 0, 0, 0};

    for (int i = 0; i < 4; i++) {
        unsigned __int128 carry = 0;
        for (int j = 0; j < 4; j++) {
            carry += (unsigned __int128)a[i] * b[j] + T[j];
            T[j] = (uint64_t)carry;
            carry >>= 64;
        }
        uint64_t c2 = (uint64_t)carry;
        carry = (unsigned __int128)T[4] + c2;
        T[4] = (uint64_t)carry;

        uint64_t m = T[0] * inv;
        carry = (unsigned __int128)T[0] + (unsigned __int128)m * p[0];
        carry >>= 64;
        for (int j = 1; j < 4; j++) {
            carry += (unsigned __int128)T[j] + (unsigned __int128)m * p[j];
            T[j - 1] = (uint64_t)carry;
            carry >>= 64;
        }
        carry += T[4];
        T[3] = (uint64_t)carry;
        T[4] = (uint64_t)(carry >> 64);
    }

    result[0] = T[0]; result[1] = T[1]; result[2] = T[2]; result[3] = T[3];

    uint64_t tmp[4];
    uint64_t borrow = host_sub_256(tmp, result, p);
    if (borrow == 0) {
        result[0] = tmp[0]; result[1] = tmp[1];
        result[2] = tmp[2]; result[3] = tmp[3];
    }
}

#endif

} // namespace bn254
