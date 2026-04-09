#pragma once

#include "curve/g1.cuh"
#include "curve/g2.cuh"
#include "field/fp12.cuh"

// ============================================================
// Optimal Ate Pairing for BN254
//
// e: G1 × G2 → GT ⊂ Fp12*
//
// The pairing consists of:
// 1. Miller loop: iterates over bits of the 6x+2 parameter
// 2. Final exponentiation: raises result to (p^12 - 1)/r
//
// BN254 parameter: x = 4965661367192848881
// |6x + 2| = 29793968202426332228 = 0x19D797039BE763BA8
// 65 bits, used in Miller loop iteration
//
// Verification equation for Groth16:
//   e(A, B) == e(α, β) · e(L, γ) · e(C, δ)
// Equivalently:
//   e(-A, B) · e(α, β) · e(L, γ) · e(C, δ) == 1
//
// Occupancy (sm_75, GTX 1650):
//   pairing_kernel: Each Fp12 = 96 registers, plus temporaries.
//   Use 32 threads/block for pairing.
//   32 * 160 regs = 5120 regs/block → 12 blocks/SM → good occupancy.
//   For batch verification, each thread computes one pairing.
// ============================================================

namespace bn254 {

// ============================================================
// BN254 uses the signed-NAF representation of |6x + 2| in the Miller loop.
// These coefficients match arkworks' BN configuration.
// ============================================================

__host__ __device__ inline const int8_t* ate_loop_count() {
    static const int8_t kAteLoopCount[] = {
        0, 0, 0, 1, 0, 1, 0, -1, 0, 0, 1, -1, 0, 0, 1, 0, 0, 1, 1, 0, -1, 0,
        0, 1, 0, -1, 0, 0, 0, 0, 1, 1, 1, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0,
        -1, 0, 0, 1, 1, 0, 0, -1, 0, 0, 0, 1, 1, 0, -1, 0, 0, 1, 0, 1, 1,
    };
    return kAteLoopCount;
}

__host__ __device__ inline int ate_loop_count_len() {
    return 65;
}

// ============================================================
// Line function evaluation structures
//
// During the Miller loop, we compute line functions ℓ_{T,Q}(P)
// where T is a running G2 point, Q is the fixed G2 input,
// and P is the G1 input.
//
// For a doubling step (T ← 2T):
//   The tangent line at T evaluated at P.
//
// For an addition step (T ← T + Q):
//   The line through T and Q evaluated at P.
//
// Each line evaluation produces 3 Fp2 coefficients that form
// a sparse Fp12 element.
// ============================================================

struct EllCoeffs {
    Fp2 c0, c1, c2;
};

struct G2HomProjective {
    Fp2 x, y, z;

    __host__ __device__ EllCoeffs double_in_place(const Fp& two_inv) {
        Fp2 a = x * y;
        a = a.mul_fp(two_inv);
        Fp2 b = y.sqr();
        Fp2 c = z.sqr();
        Fp2 e = G2Affine::generator().x; // placeholder overwritten below
        (void)e;

        // COEFF_B for BN254 D-twist: 3 / (9 + u)
        Fp2 coeff_b(
            Fp::from_raw(
                0x3bf938e377b802a8ULL, 0x020b1b273633535dULL,
                0x26b7edf049755260ULL, 0x2514c6324384a86dULL),
            Fp::from_raw(
                0x38e7ecccd1dcff67ULL, 0x65f0b37d93ce0d3eULL,
                0xd749d0dd22ac00aaULL, 0x0141b9ce4a688d4dULL)
        );

        e = coeff_b * (c + c + c);
        Fp2 f = e + e + e;
        Fp2 g = b + f;
        g = g.mul_fp(two_inv);
        Fp2 h = (y + z).sqr() - (b + c);
        Fp2 i = e - b;
        Fp2 j = x.sqr();
        Fp2 e_square = e.sqr();

        x = a * (b - f);
        y = g.sqr() - (e_square + e_square + e_square);
        z = b * h;

        return {-h, j + j + j, i};
    }

    __host__ __device__ EllCoeffs add_in_place(const G2Affine& q) {
        Fp2 theta = y - (q.y * z);
        Fp2 lambda = x - (q.x * z);
        Fp2 c = theta.sqr();
        Fp2 d = lambda.sqr();
        Fp2 e = lambda * d;
        Fp2 f = z * c;
        Fp2 g = x * d;
        Fp2 h = e + f - (g + g);
        x = lambda * h;
        y = theta * (g - h) - (e * y);
        z = z * e;
        Fp2 j = theta * q.x - (lambda * q.y);
        return {lambda, -theta, j};
    }
};

__host__ __device__ inline G2Affine mul_by_char(const G2Affine& q) {
    const Fp2 twist_mul_by_q_x(
        Fp::from_raw(
            0xb5773b104563ab30ULL, 0x347f91c8a9aa6454ULL,
            0x7a007127242e0991ULL, 0x1956bcd8118214ecULL),
        Fp::from_raw(
            0x6e849f1ea0aa4757ULL, 0xaa1c7b6d89f89141ULL,
            0xb6e713cdfae0ca3aULL, 0x26694fbb4e82ebc3ULL)
    );
    const Fp2 twist_mul_by_q_y(
        Fp::from_raw(
            0xe4bbdd0c2936b629ULL, 0xbb30f162e133bacbULL,
            0x31a9d1b6f9645366ULL, 0x253570bea500f8ddULL),
        Fp::from_raw(
            0xa1d77ce45ffe77c7ULL, 0x07affd117826d1dbULL,
            0x6d16bd27bb7edc6bULL, 0x2c87200285defeccULL)
    );

    G2Affine out = q;
    out.x = out.x.conjugate() * twist_mul_by_q_x;
    out.y = out.y.conjugate() * twist_mul_by_q_y;
    return out;
}

// ============================================================
// Evaluate line at P (G1 affine point)
//
// The line evaluation produces a sparse Fp12 element which is
// multiplied into the accumulator f.
//
// For BN254 twist: the line coefficients interact with P as:
//   ell_val = ell_0 + ell_vv * x_P + ell_vw * y_P
// where x_P, y_P are Fp elements embedded into Fp2.
// ============================================================
__host__ __device__ inline
void mul_by_line(Fp12& f, const EllCoeffs& c,
                 const Fp& px, const Fp& py) {
    // BN254 uses a D-twist, so the line element has sparse shape:
    // c0 = (c0 * py, 0, 0), c1 = (c1 * px, c2, 0).
    Fp2 coeff_0 = c.c0.mul_fp(py);
    Fp2 coeff_1 = c.c1.mul_fp(px);
    Fp12 line(
        Fp6(coeff_0, Fp2::zero(), Fp2::zero()),
        Fp6(coeff_1, c.c2, Fp2::zero())
    );
    f = f * line;
}

// ============================================================
// Miller loop: computes the raw pairing value before final exp.
//
// f = product of line evaluations at P over the bits of |6x+2|
//
// Algorithm:
//   T ← Q
//   f ← 1
//   for i = (top_bit - 1) down to 0:
//     f ← f^2
//     f ← f * ℓ_{T,T}(P)    // doubling step
//     T ← 2T
//     if bit_i(|6x+2|) == 1:
//       f ← f * ℓ_{T,Q}(P)  // addition step
//       T ← T + Q
//
// After the main loop, for BN254 we need two extra steps:
//   Q1 = π(Q)      (Frobenius endomorphism on G2)
//   Q2 = -π²(Q)
//   f ← f * ℓ_{T,Q1}(P); T ← T + Q1
//   f ← f * ℓ_{T,Q2}(P); T ← T + Q2
// ============================================================
__host__ __device__ inline
Fp12 miller_loop(const G1Affine& P, const G2Affine& Q) {
    if (P.is_infinity() || Q.is_infinity()) return Fp12::one();

    Fp12 f = Fp12::one();

    G2HomProjective r{Q.x, Q.y, Fp2::one()};
    G2Affine neg_q(Q.x, -Q.y);
    Fp two_inv = Fp::from_uint(2).inv();
    const int8_t* loop = ate_loop_count();
    int loop_len = ate_loop_count_len();

    for (int i = loop_len - 1; i >= 1; i--) {
        if (i != loop_len - 1) {
            f = f.sqr();
        }

        EllCoeffs dc = r.double_in_place(two_inv);
        mul_by_line(f, dc, P.x, P.y);

        int bit = loop[i - 1];
        if (bit == 1) {
            EllCoeffs ac = r.add_in_place(Q);
            mul_by_line(f, ac, P.x, P.y);
        } else if (bit == -1) {
            EllCoeffs ac = r.add_in_place(neg_q);
            mul_by_line(f, ac, P.x, P.y);
        }
    }

    G2Affine q1 = mul_by_char(Q);
    G2Affine q2 = mul_by_char(q1);
    q2.y = -q2.y;

    EllCoeffs c1 = r.add_in_place(q1);
    mul_by_line(f, c1, P.x, P.y);

    EllCoeffs c2 = r.add_in_place(q2);
    mul_by_line(f, c2, P.x, P.y);

    return f;
}

// ============================================================
// Full pairing: e(P, Q) = miller_loop(P, Q)^{final_exp}
// ============================================================
__host__ __device__ inline
Fp12 pairing(const G1Affine& P, const G2Affine& Q) {
    Fp12 f = miller_loop(P, Q);
    return f.final_exponentiation();
}

// ============================================================
// Multi-pairing: computes product of pairings
//   e(P1,Q1) * e(P2,Q2) * ... * e(Pn,Qn)
//
// Optimization: share the final exponentiation across all pairs.
//   result = (Π miller_loop(Pi, Qi))^{final_exp}
// ============================================================
__host__ __device__ inline
Fp12 multi_miller_loop(const G1Affine* P, const G2Affine* Q, int n) {
    Fp12 f = Fp12::one();
    for (int k = 0; k < n; k++) {
        if (P[k].is_infinity() || Q[k].is_infinity()) continue;
        Fp12 fk = miller_loop(P[k], Q[k]);
        f = f * fk;
    }
    return f;
}

// ============================================================
// Verify pairing product equation:
//   e(P1,Q1) * e(P2,Q2) * ... * e(Pn,Qn) == 1 in GT
//
// This is the core of Groth16 verification.
// For Groth16, n=4:
//   e(-A, B) · e(α, β) · e(vk_x, γ) · e(C, δ) == 1
//
// Returns true if the equation holds.
// ============================================================
__host__ __device__ inline
bool verify_pairing_product(const G1Affine* P, const G2Affine* Q, int n) {
    Fp12 f = multi_miller_loop(P, Q, n);
    Fp12 result = f.final_exponentiation();
    return result == Fp12::one();
}

// ============================================================
// GPU kernel for batch pairing verification
//
// Occupancy (sm_75): 32 threads/block, ~160 regs/thread
// 32 * 160 = 5120 regs/block → 12 blocks/SM
// 12 * 16 SMs = 192 active blocks. For batch of 192+ pairings,
// the GPU is fully utilized.
// ============================================================
__global__ void batch_pairing_verify_kernel(
    bool* results,
    const G1Affine* P_arr,  // [n_proofs * 4] — 4 G1 points per proof
    const G2Affine* Q_arr,  // [n_proofs * 4] — 4 G2 points per proof
    int n_proofs)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_proofs) return;

    const G1Affine* P = P_arr + idx * 4;
    const G2Affine* Q = Q_arr + idx * 4;

    results[idx] = verify_pairing_product(P, Q, 4);
}

} // namespace bn254
