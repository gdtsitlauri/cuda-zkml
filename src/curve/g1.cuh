#pragma once

#include "field/fp.cuh"

// ============================================================
// G1: Points on BN254 curve y^2 = x^3 + 3 over Fp
//
// Using Jacobian coordinates (X, Y, Z) where:
//   affine (x, y) = (X/Z^2, Y/Z^3)
//   Point at infinity: Z = 0
//
// BN254 G1 generator:
//   x = 1
//   y = 2
//
// Operations:
//   point_add (mixed and full Jacobian)
//   point_dbl
//   scalar_mul (double-and-add, windowed)
//
// Occupancy (sm_75):
//   G1 point = 3 Fp = 12 uint64 = 24 registers.
//   Point addition: ~40 registers/thread.
//   256 threads/block → 10240 regs/block → 6 blocks/SM. Acceptable.
// ============================================================

namespace bn254 {

struct G1Affine {
    Fp x, y;
    bool infinity;

    __host__ __device__ G1Affine() : x(), y(), infinity(true) {}
    __host__ __device__ G1Affine(const Fp& _x, const Fp& _y)
        : x(_x), y(_y), infinity(false) {}

    __host__ __device__ bool is_infinity() const { return infinity; }

    // BN254 G1 generator
    __host__ __device__ static G1Affine generator() {
        return G1Affine(Fp::from_uint(1), Fp::from_uint(2));
    }

    __host__ __device__ bool operator==(const G1Affine& other) const {
        if (infinity && other.infinity) return true;
        if (infinity != other.infinity) return false;
        return x == other.x && y == other.y;
    }

    // Check if point is on curve: y^2 = x^3 + 3
    __host__ __device__ bool is_on_curve() const {
        if (infinity) return true;
        Fp y2 = y.sqr();
        Fp x3 = x.sqr() * x;
        Fp b = Fp::from_uint(3);
        return y2 == x3 + b;
    }
};

struct G1Jacobian {
    Fp x, y, z;

    __host__ __device__ G1Jacobian() : x(Fp::zero()), y(Fp::one()), z(Fp::zero()) {}
    __host__ __device__ G1Jacobian(const Fp& _x, const Fp& _y, const Fp& _z)
        : x(_x), y(_y), z(_z) {}

    __host__ __device__ bool is_infinity() const { return z.is_zero(); }

    __host__ __device__ static G1Jacobian identity() {
        return G1Jacobian();
    }

    __host__ __device__ static G1Jacobian from_affine(const G1Affine& p) {
        if (p.is_infinity()) return identity();
        return G1Jacobian(p.x, p.y, Fp::one());
    }

    __host__ __device__ G1Affine to_affine() const {
        if (is_infinity()) return G1Affine();
        Fp z_inv = z.inv();
        Fp z_inv2 = z_inv.sqr();
        Fp z_inv3 = z_inv2 * z_inv;
        return G1Affine(x * z_inv2, y * z_inv3);
    }

    // Point doubling in Jacobian coordinates
    // Cost: 4M + 6S + 1*a + 7add (a=0 for BN254)
    // Reference: "dbl-2009-l" from EFD
    __host__ __device__ G1Jacobian dbl() const {
        if (is_infinity()) return *this;

        Fp A = x.sqr();           // A = X^2
        Fp B = y.sqr();           // B = Y^2
        Fp C = B.sqr();           // C = B^2

        // D = 2*((X+B)^2 - A - C)
        Fp D = (x + B).sqr() - A - C;
        D = D + D;

        Fp E = A + A + A;         // E = 3*A (since a=0 for BN254)

        Fp F = E.sqr();           // F = E^2

        // X3 = F - 2*D
        Fp X3 = F - D - D;

        // Y3 = E*(D - X3) - 8*C
        Fp C8 = C + C;            // 2C
        C8 = C8 + C8;             // 4C
        C8 = C8 + C8;             // 8C
        Fp Y3 = E * (D - X3) - C8;

        // Z3 = 2*Y*Z
        Fp Z3 = (y + z).sqr() - B - z.sqr();

        return G1Jacobian(X3, Y3, Z3);
    }

    // Point addition (full Jacobian + Jacobian)
    // Cost: 12M + 4S + 7add
    __host__ __device__ G1Jacobian operator+(const G1Jacobian& other) const {
        if (is_infinity()) return other;
        if (other.is_infinity()) return *this;

        Fp Z1Z1 = z.sqr();
        Fp Z2Z2 = other.z.sqr();

        Fp U1 = x * Z2Z2;
        Fp U2 = other.x * Z1Z1;

        Fp S1 = y * other.z * Z2Z2;
        Fp S2 = other.y * z * Z1Z1;

        Fp H = U2 - U1;
        Fp R = S2 - S1;

        if (H.is_zero()) {
            if (R.is_zero()) {
                return dbl(); // Same point
            }
            return identity(); // Point + (-Point)
        }

        Fp HH = H.sqr();
        Fp HHH = HH * H;

        Fp V = U1 * HH;

        // X3 = R^2 - HHH - 2*V
        Fp X3 = R.sqr() - HHH - V - V;

        // Y3 = R*(V - X3) - S1*HHH
        Fp Y3 = R * (V - X3) - S1 * HHH;

        // Z3 = Z1 * Z2 * H  (for R = S2 - S1 formulation)
        Fp Z3 = z * other.z * H;

        return G1Jacobian(X3, Y3, Z3);
    }

    // Mixed addition: Jacobian + Affine (saves some multiplications)
    // Cost: 8M + 3S + 7add (vs 12M + 4S for full)
    __host__ __device__ G1Jacobian add_affine(const G1Affine& other) const {
        if (other.is_infinity()) return *this;
        if (is_infinity()) return from_affine(other);

        Fp Z1Z1 = z.sqr();
        Fp U2 = other.x * Z1Z1;
        Fp S2 = other.y * z * Z1Z1;

        Fp H = U2 - x;
        Fp R = S2 - y;

        if (H.is_zero()) {
            if (R.is_zero()) return dbl();
            return identity();
        }

        Fp HH = H.sqr();
        Fp HHH = HH * H;
        Fp V = x * HH;

        Fp X3 = R.sqr() - HHH - V - V;
        Fp Y3 = R * (V - X3) - y * HHH;
        // Z3 = Z1 * H  (for R = S2 - Y1 formulation)
        Fp Z3 = z * H;

        return G1Jacobian(X3, Y3, Z3);
    }

    // Negation
    __host__ __device__ G1Jacobian operator-() const {
        return G1Jacobian(x, -y, z);
    }

    // Scalar multiplication (double-and-add)
    __host__ __device__ G1Jacobian scalar_mul(const Fr& scalar) const {
        G1Jacobian result = identity();
        G1Jacobian base = *this;

        // Convert scalar to standard form for bit iteration
        uint64_t s[4];
        scalar.to_standard(s);

        // Find highest bit
        int top_bit = -1;
        for (int i = 3; i >= 0; i--) {
            if (s[i] != 0) {
                #ifdef __CUDA_ARCH__
                top_bit = i * 64 + (63 - __clzll(s[i]));
                #else
                uint64_t v = s[i];
                int clz = 0;
                if (v == 0) clz = 64;
                else {
                    uint64_t tmp = v;
                    if ((tmp & 0xFFFFFFFF00000000ULL) == 0) { clz += 32; tmp <<= 32; }
                    if ((tmp & 0xFFFF000000000000ULL) == 0) { clz += 16; tmp <<= 16; }
                    if ((tmp & 0xFF00000000000000ULL) == 0) { clz += 8;  tmp <<= 8; }
                    if ((tmp & 0xF000000000000000ULL) == 0) { clz += 4;  tmp <<= 4; }
                    if ((tmp & 0xC000000000000000ULL) == 0) { clz += 2;  tmp <<= 2; }
                    if ((tmp & 0x8000000000000000ULL) == 0) { clz += 1; }
                }
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

    __host__ __device__ bool operator==(const G1Jacobian& other) const {
        if (is_infinity() && other.is_infinity()) return true;
        if (is_infinity() != other.is_infinity()) return false;
        // Compare in affine: X1*Z2^2 == X2*Z1^2 and Y1*Z2^3 == Y2*Z1^3
        Fp z1sq = z.sqr();
        Fp z2sq = other.z.sqr();
        if (x * z2sq != other.x * z1sq) return false;
        if (y * z2sq * other.z != other.y * z1sq * z) return false;
        return true;
    }
};

// Host-side kernel launch wrappers (declared in g1.cu)
void g1_scalar_mul_gpu(G1Jacobian* d_out, const G1Jacobian* d_points,
                        const Fr* d_scalars, int n);

} // namespace bn254
