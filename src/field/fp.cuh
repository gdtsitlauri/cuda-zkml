#pragma once

#include "common.cuh"
#include "field/montgomery.cuh"
#include "field/uint128_compat.cuh"

// ============================================================
// BN254 Base Field Fp
//
// p = 21888242871839275222246405745257275088696311157297823662689037894645226208583
//
// All elements stored in Montgomery form: a is stored as aR mod p
// where R = 2^256.
//
// Operations:
//   add, sub, mul, sqr, inv, pow, neg, is_zero, equals
//   to_montgomery, from_montgomery
//
// Thread safety: all operations are pure functions on registers,
// safe for massively parallel CUDA execution.
//
// Occupancy (sm_75): Fp uses 4x uint64 = 8 registers for storage.
// Montgomery mul uses ~20 additional registers.
// With 256 threads/block: ~28 regs/thread * 256 = 7168 regs/block.
// sm_75 has 65536 regs/SM → up to 9 blocks/SM = good occupancy.
// ============================================================

namespace bn254 {

struct Fp {
    uint64_t val[4]; // Montgomery form, little-endian limbs

    // ---- Constructors ----

    __host__ __device__ Fp() : val{0, 0, 0, 0} {}

    __host__ __device__ Fp(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3)
        : val{v0, v1, v2, v3} {}

    // Construct from a small integer (converts to Montgomery form)
    __host__ __device__ static Fp from_uint(uint64_t v) {
        Fp result;
        uint64_t plain[4] = {v, 0, 0, 0};
        #ifdef __CUDA_ARCH__
        to_montgomery(result.val, plain);
        #else
        // Host path: use R2_MOD_P constant directly
        const uint64_t r2[4] = {
            0xF32CFC5B538AFA89ULL, 0xB5E71911D44501FBULL,
            0x47AB1EFF0A417FF6ULL, 0x06D89F71CAB8351FULL
        };
        host_montgomery_mul(result.val, plain, r2);
        #endif
        return result;
    }

    // Construct from raw Montgomery limbs (no conversion)
    __host__ __device__ static Fp from_raw(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3) {
        return Fp(v0, v1, v2, v3);
    }

    // ---- Constants ----

    __host__ __device__ static Fp zero() { return Fp(); }

    __host__ __device__ static Fp one() {
        // 1 in Montgomery form = R mod p
        return Fp(0xD35D438DC58F0D9DULL, 0x0A78EB28F5C70B3DULL,
                   0x666EA36F7879462CULL, 0x0E0A77C19A07DF2FULL);
    }

    __host__ __device__ static Fp r_squared() {
        return Fp(0xF32CFC5B538AFA89ULL, 0xB5E71911D44501FBULL,
                   0x47AB1EFF0A417FF6ULL, 0x06D89F71CAB8351FULL);
    }

    // The prime modulus p (not in Montgomery form)
    __host__ __device__ static void get_p(uint64_t* p) {
        p[0] = 0x3C208C16D87CFD47ULL;
        p[1] = 0x97816A916871CA8DULL;
        p[2] = 0xB85045B68181585DULL;
        p[3] = 0x30644E72E131A029ULL;
    }

    // ---- Arithmetic ----
    // All arithmetic uses portable adc64/sbb64 helpers from uint128_compat.cuh
    // to support both MSVC (Windows) and GCC/Clang host compilers.

    // Modular addition: result = (a + b) mod p
    __host__ __device__ Fp operator+(const Fp& other) const {
        Fp result;
        uint64_t p[4];
        get_p(p);

        // Add with carry chain
        uint64_t c = 0;
        c = adc64(val[0], other.val[0], 0, &result.val[0]);
        c = adc64(val[1], other.val[1], c, &result.val[1]);
        c = adc64(val[2], other.val[2], c, &result.val[2]);
        uint64_t carry = adc64(val[3], other.val[3], c, &result.val[3]);

        // If result >= p, subtract p
        uint64_t tmp[4];
        uint64_t b = 0;
        b = sbb64(result.val[0], p[0], 0, &tmp[0]);
        b = sbb64(result.val[1], p[1], b, &tmp[1]);
        b = sbb64(result.val[2], p[2], b, &tmp[2]);
        b = sbb64(result.val[3], p[3], b, &tmp[3]);

        if (carry || b == 0) {
            result.val[0] = tmp[0]; result.val[1] = tmp[1];
            result.val[2] = tmp[2]; result.val[3] = tmp[3];
        }

        return result;
    }

    __host__ __device__ Fp& operator+=(const Fp& other) {
        *this = *this + other;
        return *this;
    }

    // Modular subtraction: result = (a - b) mod p
    __host__ __device__ Fp operator-(const Fp& other) const {
        Fp result;
        uint64_t p[4];
        get_p(p);

        uint64_t borrow = 0;
        borrow = sbb64(val[0], other.val[0], 0, &result.val[0]);
        borrow = sbb64(val[1], other.val[1], borrow, &result.val[1]);
        borrow = sbb64(val[2], other.val[2], borrow, &result.val[2]);
        borrow = sbb64(val[3], other.val[3], borrow, &result.val[3]);

        // If borrow, add p
        if (borrow) {
            uint64_t c = 0;
            c = adc64(result.val[0], p[0], 0, &result.val[0]);
            c = adc64(result.val[1], p[1], c, &result.val[1]);
            c = adc64(result.val[2], p[2], c, &result.val[2]);
            adc64(result.val[3], p[3], c, &result.val[3]);
        }

        return result;
    }

    __host__ __device__ Fp& operator-=(const Fp& other) {
        *this = *this - other;
        return *this;
    }

    // Modular negation: result = -a mod p = p - a (or 0 if a == 0)
    __host__ __device__ Fp operator-() const {
        if (is_zero()) return *this;
        Fp result;
        uint64_t p[4];
        get_p(p);
        uint64_t borrow = 0;
        borrow = sbb64(p[0], val[0], 0, &result.val[0]);
        borrow = sbb64(p[1], val[1], borrow, &result.val[1]);
        borrow = sbb64(p[2], val[2], borrow, &result.val[2]);
        sbb64(p[3], val[3], borrow, &result.val[3]);
        return result;
    }

    // Montgomery multiplication
    __host__ __device__ Fp operator*(const Fp& other) const {
        Fp result;
        #ifdef __CUDA_ARCH__
        montgomery_mul_cios(result.val, val, other.val);
        #else
        host_montgomery_mul(result.val, val, other.val);
        #endif
        return result;
    }

    __host__ __device__ Fp& operator*=(const Fp& other) {
        *this = *this * other;
        return *this;
    }

    // Squaring (currently implemented via multiplication; can be specialized)
    __host__ __device__ Fp sqr() const {
        return *this * *this;
    }

    // Exponentiation by squaring: result = base^exp mod p
    // exp is given as 4 limbs (not Montgomery form, just a scalar)
    __host__ __device__ Fp pow(const uint64_t exp[4]) const {
        Fp result = one();
        Fp base = *this;

        // Find highest bit
        int top_bit = -1;
        for (int i = 3; i >= 0; i--) {
            if (exp[i] != 0) {
                uint64_t v = exp[i];
                int clz = 0;
                #ifdef __CUDA_ARCH__
                clz = __clzll(v);
                #else
                if (v == 0) clz = 64;
                else {
                    clz = 0;
                    uint64_t tmp = v;
                    if ((tmp & 0xFFFFFFFF00000000ULL) == 0) { clz += 32; tmp <<= 32; }
                    if ((tmp & 0xFFFF000000000000ULL) == 0) { clz += 16; tmp <<= 16; }
                    if ((tmp & 0xFF00000000000000ULL) == 0) { clz += 8;  tmp <<= 8; }
                    if ((tmp & 0xF000000000000000ULL) == 0) { clz += 4;  tmp <<= 4; }
                    if ((tmp & 0xC000000000000000ULL) == 0) { clz += 2;  tmp <<= 2; }
                    if ((tmp & 0x8000000000000000ULL) == 0) { clz += 1; }
                }
                #endif
                top_bit = i * 64 + (63 - clz);
                break;
            }
        }

        if (top_bit < 0) return one(); // exp == 0

        for (int bit = top_bit; bit >= 0; bit--) {
            result = result.sqr();
            int limb = bit / 64;
            int pos = bit % 64;
            if ((exp[limb] >> pos) & 1) {
                result = result * base;
            }
        }

        return result;
    }

    // Power with uint64_t exponent
    __host__ __device__ Fp pow_u64(uint64_t exp) const {
        uint64_t e[4] = {exp, 0, 0, 0};
        return pow(e);
    }

    // Modular inverse using Fermat's little theorem: a^{-1} = a^{p-2} mod p
    __host__ __device__ Fp inv() const {
        // p - 2
        uint64_t p_minus_2[4] = {
            0x3C208C16D87CFD45ULL,  // p[0] - 2
            0x97816A916871CA8DULL,
            0xB85045B68181585DULL,
            0x30644E72E131A029ULL
        };
        return pow(p_minus_2);
    }

    // Division
    __host__ __device__ Fp operator/(const Fp& other) const {
        return *this * other.inv();
    }

    // ---- Comparison ----

    __host__ __device__ bool operator==(const Fp& other) const {
        return val[0] == other.val[0] && val[1] == other.val[1] &&
               val[2] == other.val[2] && val[3] == other.val[3];
    }

    __host__ __device__ bool operator!=(const Fp& other) const {
        return !(*this == other);
    }

    __host__ __device__ bool is_zero() const {
        return val[0] == 0 && val[1] == 0 && val[2] == 0 && val[3] == 0;
    }

    // ---- Conversion ----

    // Convert from standard form to Montgomery form
    __host__ __device__ static Fp from_standard(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3) {
        Fp plain(v0, v1, v2, v3);
        Fp result;
        #ifdef __CUDA_ARCH__
        montgomery_mul_cios(result.val, plain.val, R2_MOD_P);
        #else
        const uint64_t r2[4] = {
            0xF32CFC5B538AFA89ULL, 0xB5E71911D44501FBULL,
            0x47AB1EFF0A417FF6ULL, 0x06D89F71CAB8351FULL
        };
        host_montgomery_mul(result.val, plain.val, r2);
        #endif
        return result;
    }

    // Convert from Montgomery form to standard form
    __host__ __device__ void to_standard(uint64_t* out) const {
        uint64_t one_val[4] = {1, 0, 0, 0};
        #ifdef __CUDA_ARCH__
        montgomery_mul_cios(out, val, one_val);
        #else
        host_montgomery_mul(out, val, one_val);
        #endif
    }

    // ---- Debug ----

    void print(const char* label = "") const {
        uint64_t std_form[4];
        to_standard(std_form);
        printf("%s0x%016llx%016llx%016llx%016llx\n", label,
               (unsigned long long)std_form[3], (unsigned long long)std_form[2],
               (unsigned long long)std_form[1], (unsigned long long)std_form[0]);
    }

    void print_mont(const char* label = "") const {
        printf("%s[mont] 0x%016llx%016llx%016llx%016llx\n", label,
               (unsigned long long)val[3], (unsigned long long)val[2],
               (unsigned long long)val[1], (unsigned long long)val[0]);
    }
};

// ============================================================
// BN254 scalar field Fr (for curve scalar operations)
// r = 21888242871839275222246405745257275088548364400416034343698204186575808495617
// In hex: 0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001
// ============================================================

struct Fr {
    uint64_t val[4];

    __host__ __device__ Fr() : val{0, 0, 0, 0} {}
    __host__ __device__ Fr(uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3)
        : val{v0, v1, v2, v3} {}

    __host__ __device__ static void get_r(uint64_t* r) {
        r[0] = 0x43E1F593F0000001ULL;
        r[1] = 0x2833E84879B97091ULL;
        r[2] = 0xB85045B68181585DULL;
        r[3] = 0x30644E72E131A029ULL;
    }

    // R_INV for Fr: -r^{-1} mod 2^64
    static constexpr uint64_t FR_INV = 0xC2E1F593EFFFFFFFULL;

    // R mod r
    __host__ __device__ static Fr one() {
        return Fr(0xAC96341C4FFFFFFBULL, 0x36FC76959F60CD29ULL,
                  0x666EA36F7879462EULL, 0x0E0A77C19A07DF2FULL);
    }

    // R^2 mod r
    __host__ __device__ static Fr r_squared() {
        return Fr(0x1BB8E645AE216DA7ULL, 0x53FE3AB1E35C59E3ULL,
                  0x8C49833D53BB8085ULL, 0x0216D0B17F4E44A5ULL);
    }

    __host__ __device__ static Fr zero() { return Fr(); }

    __host__ __device__ bool is_zero() const {
        return val[0] == 0 && val[1] == 0 && val[2] == 0 && val[3] == 0;
    }

    __host__ __device__ bool operator==(const Fr& other) const {
        return val[0] == other.val[0] && val[1] == other.val[1] &&
               val[2] == other.val[2] && val[3] == other.val[3];
    }

    // Montgomery mul for Fr using portable helpers
    __host__ __device__ static void mont_mul_fr(uint64_t* result, const uint64_t* a, const uint64_t* b) {
        uint64_t r[4];
        get_r(r);
        const uint64_t inv = FR_INV;
        uint64_t T[5] = {0, 0, 0, 0, 0};

        for (int i = 0; i < 4; i++) {
            // Step 1: T = T + a[i] * b
            uint64_t carry_lo = 0;
            for (int j = 0; j < 4; j++) {
                uint64_t hi;
                uint64_t lo = mul64_full(a[i], b[j], &hi);
                // acc = T[j] + lo + carry
                uint64_t t;
                uint64_t c1 = adc64(T[j], lo, 0, &t);
                uint64_t c2 = adc64(t, carry_lo, 0, &T[j]);
                carry_lo = hi + c1 + c2;
            }
            uint64_t c = adc64(T[4], carry_lo, 0, &T[4]);
            (void)c;

            // Step 2: m = T[0] * inv mod 2^64
            uint64_t m = T[0] * inv;

            // Step 3: T = (T + m * r) >> 64
            carry_lo = 0;
            {
                uint64_t hi;
                uint64_t lo = mul64_full(m, r[0], &hi);
                uint64_t t;
                adc64(T[0], lo, 0, &t);
                carry_lo = hi + (t < T[0] || t < lo ? 1 : 0);
                // Actually: we just need (T[0] + m*r[0]) >> 64
                uint64_t sum_lo;
                uint64_t sum_carry = adc64(T[0], lo, 0, &sum_lo);
                carry_lo = hi + sum_carry;
            }
            for (int j = 1; j < 4; j++) {
                uint64_t hi;
                uint64_t lo = mul64_full(m, r[j], &hi);
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

        // Final reduction
        uint64_t tmp[4];
        uint64_t borrow = 0;
        borrow = sbb64(result[0], r[0], 0, &tmp[0]);
        borrow = sbb64(result[1], r[1], borrow, &tmp[1]);
        borrow = sbb64(result[2], r[2], borrow, &tmp[2]);
        borrow = sbb64(result[3], r[3], borrow, &tmp[3]);
        if (borrow == 0) {
            result[0] = tmp[0]; result[1] = tmp[1]; result[2] = tmp[2]; result[3] = tmp[3];
        }
    }

    __host__ __device__ Fr operator*(const Fr& other) const {
        Fr result;
        mont_mul_fr(result.val, val, other.val);
        return result;
    }

    __host__ __device__ Fr operator+(const Fr& other) const {
        Fr result;
        uint64_t r[4]; get_r(r);
        uint64_t c = 0;
        c = adc64(val[0], other.val[0], 0, &result.val[0]);
        c = adc64(val[1], other.val[1], c, &result.val[1]);
        c = adc64(val[2], other.val[2], c, &result.val[2]);
        uint64_t carry = adc64(val[3], other.val[3], c, &result.val[3]);

        uint64_t tmp[4];
        uint64_t b = 0;
        b = sbb64(result.val[0], r[0], 0, &tmp[0]);
        b = sbb64(result.val[1], r[1], b, &tmp[1]);
        b = sbb64(result.val[2], r[2], b, &tmp[2]);
        b = sbb64(result.val[3], r[3], b, &tmp[3]);
        if (carry || b == 0) {
            result.val[0] = tmp[0]; result.val[1] = tmp[1];
            result.val[2] = tmp[2]; result.val[3] = tmp[3];
        }
        return result;
    }

    __host__ __device__ Fr operator-(const Fr& other) const {
        Fr result;
        uint64_t r[4]; get_r(r);
        uint64_t borrow = 0;
        borrow = sbb64(val[0], other.val[0], 0, &result.val[0]);
        borrow = sbb64(val[1], other.val[1], borrow, &result.val[1]);
        borrow = sbb64(val[2], other.val[2], borrow, &result.val[2]);
        borrow = sbb64(val[3], other.val[3], borrow, &result.val[3]);
        if (borrow) {
            uint64_t c = 0;
            c = adc64(result.val[0], r[0], 0, &result.val[0]);
            c = adc64(result.val[1], r[1], c, &result.val[1]);
            c = adc64(result.val[2], r[2], c, &result.val[2]);
            adc64(result.val[3], r[3], c, &result.val[3]);
        }
        return result;
    }

    __host__ __device__ Fr operator-() const {
        if (is_zero()) return *this;
        Fr result;
        uint64_t r[4]; get_r(r);
        uint64_t borrow = 0;
        borrow = sbb64(r[0], val[0], 0, &result.val[0]);
        borrow = sbb64(r[1], val[1], borrow, &result.val[1]);
        borrow = sbb64(r[2], val[2], borrow, &result.val[2]);
        sbb64(r[3], val[3], borrow, &result.val[3]);
        return result;
    }

    __host__ __device__ static Fr from_uint(uint64_t v) {
        Fr plain(v, 0, 0, 0);
        Fr result;
        Fr r2 = r_squared();
        mont_mul_fr(result.val, plain.val, r2.val);
        return result;
    }

    __host__ __device__ Fr sqr() const {
        return *this * *this;
    }

    __host__ __device__ Fr pow(const uint64_t exp[4]) const {
        Fr result = one();
        Fr base = *this;

        int top_bit = -1;
        for (int i = 3; i >= 0; i--) {
            if (exp[i] != 0) {
                uint64_t v = exp[i];
                int clz = 0;
                #ifdef __CUDA_ARCH__
                clz = __clzll(v);
                #else
                if (v == 0) clz = 64;
                else {
                    clz = 0;
                    uint64_t tmp = v;
                    if ((tmp & 0xFFFFFFFF00000000ULL) == 0) { clz += 32; tmp <<= 32; }
                    if ((tmp & 0xFFFF000000000000ULL) == 0) { clz += 16; tmp <<= 16; }
                    if ((tmp & 0xFF00000000000000ULL) == 0) { clz += 8;  tmp <<= 8; }
                    if ((tmp & 0xF000000000000000ULL) == 0) { clz += 4;  tmp <<= 4; }
                    if ((tmp & 0xC000000000000000ULL) == 0) { clz += 2;  tmp <<= 2; }
                    if ((tmp & 0x8000000000000000ULL) == 0) { clz += 1; }
                }
                #endif
                top_bit = i * 64 + (63 - clz);
                break;
            }
        }

        if (top_bit < 0) return one();

        for (int bit = top_bit; bit >= 0; bit--) {
            result = result.sqr();
            int limb = bit / 64;
            int pos = bit % 64;
            if ((exp[limb] >> pos) & 1) {
                result = result * base;
            }
        }

        return result;
    }

    __host__ __device__ Fr pow_u64(uint64_t exp) const {
        uint64_t e[4] = {exp, 0, 0, 0};
        return pow(e);
    }

    __host__ __device__ Fr inv() const {
        uint64_t r_minus_2[4] = {
            0x43E1F593EFFFFFFFULL,
            0x2833E84879B97091ULL,
            0xB85045B68181585DULL,
            0x30644E72E131A029ULL
        };
        return pow(r_minus_2);
    }

    __host__ __device__ Fr operator/(const Fr& other) const {
        return *this * other.inv();
    }

    __host__ __device__ void to_standard(uint64_t* out) const {
        uint64_t one_val[4] = {1, 0, 0, 0};
        mont_mul_fr(out, val, one_val);
    }

    __host__ __device__ bool bit(int i) const {
        // Need to convert to standard form to read bits of the actual scalar
        uint64_t std[4];
        to_standard(std);
        int limb = i / 64;
        int pos = i % 64;
        if (limb >= 4) return false;
        return (std[limb] >> pos) & 1;
    }

    // Bit access on raw Montgomery representation (for MSM)
    __host__ __device__ bool raw_bit(int i) const {
        int limb = i / 64;
        int pos = i % 64;
        if (limb >= 4) return false;
        return (val[limb] >> pos) & 1;
    }
};

} // namespace bn254
