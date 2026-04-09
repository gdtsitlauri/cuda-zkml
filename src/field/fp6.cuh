#pragma once

#include "field/fp2.cuh"

// ============================================================
// Fp6 = Fp2[v] / (v^3 - ξ) where ξ = 9 + u ∈ Fp2
// Elements: a + bv + cv^2 where a, b, c ∈ Fp2
//
// This is part of the tower for BN254 pairing:
// Fp → Fp2 → Fp6 → Fp12
//
// Occupancy (sm_75): Each Fp6 = 3 Fp2 = 6 Fp = 24 uint64 = 48 registers.
// Fp6 mul uses ~150 registers. With 128 threads/block → may limit
// to 2-3 blocks/SM. For pairing we accept lower occupancy.
// ============================================================

namespace bn254 {

struct Fp6 {
    Fp2 c0, c1, c2; // c0 + c1*v + c2*v^2

    __host__ __device__ Fp6() {}
    __host__ __device__ Fp6(const Fp2& a, const Fp2& b, const Fp2& c)
        : c0(a), c1(b), c2(c) {}

    __host__ __device__ static Fp6 zero() {
        return Fp6(Fp2::zero(), Fp2::zero(), Fp2::zero());
    }

    __host__ __device__ static Fp6 one() {
        return Fp6(Fp2::one(), Fp2::zero(), Fp2::zero());
    }

    __host__ __device__ bool is_zero() const {
        return c0.is_zero() && c1.is_zero() && c2.is_zero();
    }

    __host__ __device__ bool operator==(const Fp6& other) const {
        return c0 == other.c0 && c1 == other.c1 && c2 == other.c2;
    }

    __host__ __device__ Fp6 operator+(const Fp6& other) const {
        return Fp6(c0 + other.c0, c1 + other.c1, c2 + other.c2);
    }

    __host__ __device__ Fp6 operator-(const Fp6& other) const {
        return Fp6(c0 - other.c0, c1 - other.c1, c2 - other.c2);
    }

    __host__ __device__ Fp6 operator-() const {
        return Fp6(-c0, -c1, -c2);
    }

    // Fp6 multiplication using Karatsuba-like approach
    // (a0 + a1*v + a2*v^2)(b0 + b1*v + b2*v^2)
    // where v^3 = ξ (non-residue in Fp2)
    __host__ __device__ Fp6 operator*(const Fp6& other) const {
        // Chung-Hasan SQR3-derived multiplication (6 Fp2 muls instead of 9)
        Fp2 a0b0 = c0 * other.c0;
        Fp2 a1b1 = c1 * other.c1;
        Fp2 a2b2 = c2 * other.c2;

        // r0 = a0*b0 + ξ*((a1+a2)(b1+b2) - a1*b1 - a2*b2)
        Fp2 t0 = (c1 + c2) * (other.c1 + other.c2) - a1b1 - a2b2;
        Fp2 r0 = a0b0 + t0.mul_by_nonresidue();

        // r1 = (a0+a1)(b0+b1) - a0*b0 - a1*b1 + ξ*a2*b2
        Fp2 r1 = (c0 + c1) * (other.c0 + other.c1) - a0b0 - a1b1 + a2b2.mul_by_nonresidue();

        // r2 = (a0+a2)(b0+b2) - a0*b0 - a2*b2 + a1*b1
        Fp2 r2 = (c0 + c2) * (other.c0 + other.c2) - a0b0 - a2b2 + a1b1;

        return Fp6(r0, r1, r2);
    }

    __host__ __device__ Fp6& operator*=(const Fp6& other) {
        *this = *this * other;
        return *this;
    }

    // Squaring
    __host__ __device__ Fp6 sqr() const {
        Fp2 s0 = c0.sqr();
        Fp2 ab = c0 * c1;
        Fp2 s1 = ab + ab; // 2*a0*a1
        Fp2 s2 = (c0 - c1 + c2).sqr();
        Fp2 bc = c1 * c2;
        Fp2 s3 = bc + bc; // 2*a1*a2
        Fp2 s4 = c2.sqr();

        // r0 = s0 + ξ*s3
        Fp2 r0 = s0 + s3.mul_by_nonresidue();
        // r1 = s1 + ξ*s4
        Fp2 r1 = s1 + s4.mul_by_nonresidue();
        // r2 = s1 + s2 + s3 - s0 - s4
        Fp2 r2 = s1 + s2 + s3 - s0 - s4;

        return Fp6(r0, r1, r2);
    }

    // Inverse in Fp6
    __host__ __device__ Fp6 inv() const {
        Fp2 c0s = c0.sqr();
        Fp2 c1s = c1.sqr();
        Fp2 c2s = c2.sqr();
        Fp2 c0c1 = c0 * c1;
        Fp2 c0c2 = c0 * c2;
        Fp2 c1c2 = c1 * c2;

        // Compute cofactors
        Fp2 A = c0s - c1c2.mul_by_nonresidue();
        Fp2 B = c2s.mul_by_nonresidue() - c0c1;
        Fp2 C = c1s - c0c2;

        Fp2 F = c0 * A + (c2 * B + c1 * C).mul_by_nonresidue();
        Fp2 F_inv = F.inv();

        return Fp6(A * F_inv, B * F_inv, C * F_inv);
    }

    // Multiply by scalar in Fp2
    __host__ __device__ Fp6 mul_fp2(const Fp2& s) const {
        return Fp6(c0 * s, c1 * s, c2 * s);
    }

    // Multiply by non-residue for Fp12: multiply by v
    // (c0 + c1*v + c2*v^2) * v = c2*ξ + c0*v + c1*v^2
    __host__ __device__ Fp6 mul_by_v() const {
        return Fp6(c2.mul_by_nonresidue(), c0, c1);
    }
};

} // namespace bn254
