#pragma once

#include "field/fp2.cuh"
#include "field/fp.cuh"

// ============================================================
// G2: Points on BN254 twist curve y^2 = x^3 + b' over Fp2
// where b' = 3/(9+u) = 3/(9+u) in Fp2
//
// BN254 G2 generator (from EIP-197):
// x = (10857046999023057135944570762232829481370756359578518086990519993285655852781,
//      11559732032986387107991004021392285783925812861821192530917403151452391805634)
// y = (8495653923123431417604973247489272438418190587263600148770280649306958101930,
//      4082367875863433681332203403145435568316851327593401208105741076214120093531)
//
// Jacobian coordinates, same formulas as G1 but over Fp2.
//
// Occupancy (sm_75): G2 point = 3 Fp2 = 6 Fp = 24 uint64 = 48 registers.
// G2 operations are ~2-3x heavier than G1 due to Fp2 arithmetic.
// Use 64-128 threads/block for G2 kernels.
// ============================================================

namespace bn254 {

struct G2Affine {
    Fp2 x, y;
    bool infinity;

    __host__ __device__ G2Affine() : x(), y(), infinity(true) {}
    __host__ __device__ G2Affine(const Fp2& _x, const Fp2& _y)
        : x(_x), y(_y), infinity(false) {}

    __host__ __device__ bool is_infinity() const { return infinity; }

    __host__ __device__ static Fp2 curve_b() {
        return Fp2(
            Fp::from_raw(
                0x3bf938e377b802a8ULL, 0x020b1b273633535dULL,
                0x26b7edf049755260ULL, 0x2514c6324384a86dULL),
            Fp::from_raw(
                0x38e7ecccd1dcff67ULL, 0x65f0b37d93ce0d3eULL,
                0xd749d0dd22ac00aaULL, 0x0141b9ce4a688d4dULL)
        );
    }

    // BN254 G2 generator
    __host__ __device__ static G2Affine generator() {
        Fp x0 = Fp::from_raw(
            0x8e83b5d102bc2026ULL, 0xdceb1935497b0172ULL,
            0xfbb8264797811adfULL, 0x19573841af96503bULL
        );
        Fp x1 = Fp::from_raw(
            0xafb4737da84c6140ULL, 0x6043dd5a5802d8c4ULL,
            0x09e950fc52a02f86ULL, 0x14fef0833aea7b6bULL
        );
        Fp y0 = Fp::from_raw(
            0x619dfa9d886be9f6ULL, 0xfe7fd297f59e9b78ULL,
            0xff9e1a62231b7dfeULL, 0x28fd7eebae9e4206ULL
        );
        Fp y1 = Fp::from_raw(
            0x64095b56c71856eeULL, 0xdc57f922327d3cbbULL,
            0x55f935be33351076ULL, 0x0da4a0e693fd6482ULL
        );
        return G2Affine(Fp2(x0, x1), Fp2(y0, y1));
    }

    __host__ __device__ bool operator==(const G2Affine& other) const {
        if (infinity && other.infinity) return true;
        if (infinity != other.infinity) return false;
        return x == other.x && y == other.y;
    }

    __host__ __device__ bool is_on_curve() const {
        if (infinity) return true;
        return y.sqr() == x.sqr() * x + curve_b();
    }
};

struct G2Jacobian {
    Fp2 x, y, z;

    __host__ __device__ G2Jacobian() : x(Fp2::zero()), y(Fp2::one()), z(Fp2::zero()) {}
    __host__ __device__ G2Jacobian(const Fp2& _x, const Fp2& _y, const Fp2& _z)
        : x(_x), y(_y), z(_z) {}

    __host__ __device__ bool is_infinity() const { return z.is_zero(); }

    __host__ __device__ static G2Jacobian identity() { return G2Jacobian(); }

    __host__ __device__ static G2Jacobian from_affine(const G2Affine& p) {
        if (p.is_infinity()) return identity();
        return G2Jacobian(p.x, p.y, Fp2::one());
    }

    __host__ __device__ G2Affine to_affine() const {
        if (is_infinity()) return G2Affine();
        Fp2 z_inv = z.inv();
        Fp2 z_inv2 = z_inv.sqr();
        Fp2 z_inv3 = z_inv2 * z_inv;
        return G2Affine(x * z_inv2, y * z_inv3);
    }

    __host__ __device__ static G2Affine double_affine_point(const G2Affine& p) {
        if (p.is_infinity() || p.y.is_zero()) return G2Affine();

        Fp two_fp = Fp::from_uint(2);
        Fp three_fp = Fp::from_uint(3);
        Fp2 two(two_fp, Fp::zero());
        Fp2 three(three_fp, Fp::zero());

        Fp2 lambda = (p.x.sqr() * three) / (p.y * two);
        Fp2 x3 = lambda.sqr() - p.x - p.x;
        Fp2 y3 = lambda * (p.x - x3) - p.y;
        return G2Affine(x3, y3);
    }

    __host__ __device__ static G2Affine add_affine_points(const G2Affine& p,
                                                          const G2Affine& q) {
        if (p.is_infinity()) return q;
        if (q.is_infinity()) return p;

        if (p.x == q.x) {
            if (p.y != q.y) return G2Affine();
            return double_affine_point(p);
        }

        Fp2 lambda = (q.y - p.y) / (q.x - p.x);
        Fp2 x3 = lambda.sqr() - p.x - q.x;
        Fp2 y3 = lambda * (p.x - x3) - p.y;
        return G2Affine(x3, y3);
    }

    // Point doubling. Use affine formulas for correctness on arbitrary points.
    __host__ __device__ G2Jacobian dbl() const {
        return from_affine(double_affine_point(to_affine()));
    }

    // Point addition. Use affine formulas for correctness on arbitrary points.
    __host__ __device__ G2Jacobian operator+(const G2Jacobian& other) const {
        return from_affine(add_affine_points(to_affine(), other.to_affine()));
    }

    // Mixed addition.
    __host__ __device__ G2Jacobian add_affine(const G2Affine& other) const {
        return from_affine(add_affine_points(to_affine(), other));
    }

    __host__ __device__ G2Jacobian operator-() const {
        return G2Jacobian(x, -y, z);
    }

    // Scalar multiplication
    __host__ __device__ G2Jacobian scalar_mul(const Fr& scalar) const {
        G2Jacobian result = identity();
        G2Jacobian base = *this;

        uint64_t s[4];
        scalar.to_standard(s);

        int top_bit = -1;
        for (int i = 3; i >= 0; i--) {
            if (s[i] != 0) {
                #ifdef __CUDA_ARCH__
                top_bit = i * 64 + (63 - __clzll(s[i]));
                #else
                uint64_t v = s[i];
                int clz = 0;
                uint64_t tmp = v;
                if ((tmp & 0xFFFFFFFF00000000ULL) == 0) { clz += 32; tmp <<= 32; }
                if ((tmp & 0xFFFF000000000000ULL) == 0) { clz += 16; tmp <<= 16; }
                if ((tmp & 0xFF00000000000000ULL) == 0) { clz += 8;  tmp <<= 8; }
                if ((tmp & 0xF000000000000000ULL) == 0) { clz += 4;  tmp <<= 4; }
                if ((tmp & 0xC000000000000000ULL) == 0) { clz += 2;  tmp <<= 2; }
                if ((tmp & 0x8000000000000000ULL) == 0) { clz += 1; }
                top_bit = i * 64 + (63 - clz);
                #endif
                break;
            }
        }

        if (top_bit < 0) return identity();

        for (int bit = top_bit; bit >= 0; bit--) {
            result = result.dbl();
            int limb = bit / 64;
            int pos = bit % 64;
            if ((s[limb] >> pos) & 1ULL) {
                result = result + base;
            }
        }
        return result;
    }

    __host__ __device__ bool operator==(const G2Jacobian& other) const {
        if (is_infinity() && other.is_infinity()) return true;
        if (is_infinity() != other.is_infinity()) return false;
        Fp2 z1sq = z.sqr();
        Fp2 z2sq = other.z.sqr();
        if (!(x * z2sq == other.x * z1sq)) return false;
        if (!(y * z2sq * other.z == other.y * z1sq * z)) return false;
        return true;
    }
};

void g2_scalar_mul_gpu(G2Jacobian* d_out, const G2Jacobian* d_points,
                        const Fr* d_scalars, int n);

} // namespace bn254
