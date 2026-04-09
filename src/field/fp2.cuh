#pragma once

#include "field/fp.cuh"

// ============================================================
// Fp2 = Fp[u] / (u^2 + 1)
// Elements: a + b*u where a, b ∈ Fp
// Used for G2 points on BN254 twist curve
//
// Arithmetic:
//   (a + bu)(c + du) = (ac - bd) + (ad + bc)u
//   Conjugate: a + bu → a - bu
//   Norm: (a + bu)(a - bu) = a^2 + b^2
//
// Occupancy (sm_75): Each Fp2 = 2 Fp = 8 uint64 = 16 registers.
// Fp2 mul = 3 Fp muls + some adds. ~60 registers per thread for a mul.
// With 128 threads/block → 7680 regs/block → up to 8 blocks/SM.
// ============================================================

namespace bn254 {

struct Fp2 {
    Fp c0; // real part
    Fp c1; // imaginary part (coefficient of u)

    __host__ __device__ Fp2() : c0(), c1() {}
    __host__ __device__ Fp2(const Fp& a, const Fp& b) : c0(a), c1(b) {}

    __host__ __device__ static Fp2 zero() { return Fp2(Fp::zero(), Fp::zero()); }
    __host__ __device__ static Fp2 one() { return Fp2(Fp::one(), Fp::zero()); }

    __host__ __device__ bool is_zero() const {
        return c0.is_zero() && c1.is_zero();
    }

    __host__ __device__ bool operator==(const Fp2& other) const {
        return c0 == other.c0 && c1 == other.c1;
    }

    __host__ __device__ bool operator!=(const Fp2& other) const {
        return !(*this == other);
    }

    // (a + bu) + (c + du) = (a+c) + (b+d)u
    __host__ __device__ Fp2 operator+(const Fp2& other) const {
        return Fp2(c0 + other.c0, c1 + other.c1);
    }

    __host__ __device__ Fp2& operator+=(const Fp2& other) {
        c0 += other.c0;
        c1 += other.c1;
        return *this;
    }

    // (a + bu) - (c + du) = (a-c) + (b-d)u
    __host__ __device__ Fp2 operator-(const Fp2& other) const {
        return Fp2(c0 - other.c0, c1 - other.c1);
    }

    __host__ __device__ Fp2& operator-=(const Fp2& other) {
        c0 -= other.c0;
        c1 -= other.c1;
        return *this;
    }

    __host__ __device__ Fp2 operator-() const {
        return Fp2(-c0, -c1);
    }

    // Karatsuba multiplication:
    // (a + bu)(c + du) = (ac - bd) + ((a+b)(c+d) - ac - bd)u
    // Cost: 3 Fp muls (vs 4 for naive)
    __host__ __device__ Fp2 operator*(const Fp2& other) const {
        Fp ac = c0 * other.c0;
        Fp bd = c1 * other.c1;
        Fp ab_cd = (c0 + c1) * (other.c0 + other.c1);
        return Fp2(ac - bd, ab_cd - ac - bd);
    }

    __host__ __device__ Fp2& operator*=(const Fp2& other) {
        *this = *this * other;
        return *this;
    }

    // Multiply by Fp scalar
    __host__ __device__ Fp2 mul_fp(const Fp& s) const {
        return Fp2(c0 * s, c1 * s);
    }

    // Squaring: (a + bu)^2 = (a^2 - b^2) + 2ab*u
    // Optimized: use (a+b)(a-b) for real part
    __host__ __device__ Fp2 sqr() const {
        Fp ab2 = c0 * c1;
        ab2 = ab2 + ab2; // 2ab
        Fp real = (c0 + c1) * (c0 - c1); // a^2 - b^2
        return Fp2(real, ab2);
    }

    // Conjugate: a + bu → a - bu
    __host__ __device__ Fp2 conjugate() const {
        return Fp2(c0, -c1);
    }

    // Norm: a^2 + b^2 (since u^2 = -1)
    __host__ __device__ Fp norm() const {
        return c0.sqr() + c1.sqr();
    }

    // Inverse: (a + bu)^{-1} = (a - bu) / (a^2 + b^2)
    __host__ __device__ Fp2 inv() const {
        Fp n = norm();
        Fp n_inv = n.inv();
        return Fp2(c0 * n_inv, -(c1 * n_inv));
    }

    __host__ __device__ Fp2 operator/(const Fp2& other) const {
        return *this * other.inv();
    }

    // Multiply by non-residue for Fp6 tower
    // In BN254: non-residue for Fp2 → Fp6 is (9 + u)
    // mul_by_nonresidue: multiply by (9 + u) in Fp2
    // (a + bu)(9 + u) = (9a - b) + (a + 9b)u
    __host__ __device__ Fp2 mul_by_nonresidue() const {
        Fp nine = Fp::from_uint(9);
        return Fp2(nine * c0 - c1, c0 + nine * c1);
    }

    void print(const char* label = "") const {
        printf("%s", label);
        c0.print("  c0=");
        c1.print("  c1=");
    }
};

} // namespace bn254
