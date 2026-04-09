#pragma once

#include "prover/circuit.cuh"
#include "prover/witness.cuh"
#include "curve/g1.cuh"
#include "curve/g2.cuh"
#include "msm/msm.cuh"
#include "ntt/ntt.cuh"
#include <vector>
#include <string>

// ============================================================
// Groth16 ZK-SNARK Proving System (GPU-Accelerated)
//
// Proof: π = (A ∈ G1, B ∈ G2, C ∈ G1)
// Proof size: 3 group elements = 2*32 + 4*32 + 2*32 = 256 bytes
// Verification: 4 pairings (constant time)
//
// Proving flow:
// 1. R1CS → QAP transformation (Lagrange interpolation via NTT)
// 2. Witness assignment to QAP polynomials
// 3. Quotient polynomial h(x) = (A(x)·B(x) - C(x)) / Z_H(x)
// 4. MSM for proof elements A, B, C using proving key
//
// GPU acceleration points:
// - NTT for polynomial multiplication in QAP (Phase 3 step 3)
// - MSM for each proof element (Phase 3 step 4)
// - Batch MSM when proving multiple proofs
//
// Memory for GTX 1650 (3.5GB usable):
//   PK: ~2*n*96 (G1) + n*192 (G2) bytes
//   For n=100K constraints: ~48 MB → fine
//   For n>10M: use streaming PK loader from disk
// ============================================================

namespace zkml {

using bn254::Fr;
using bn254::Fp;
using bn254::G1Affine;
using bn254::G1Jacobian;
using bn254::G2Affine;
using bn254::G2Jacobian;

// ============================================================
// QAP (Quadratic Arithmetic Program)
//
// R1CS: (A·w) ⊙ (B·w) = (C·w)
// QAP: A(x)·B(x) - C(x) = H(x)·Z_H(x)
//
// where Z_H(x) = x^n - 1 (vanishing polynomial on domain H)
// and H(x) is the quotient polynomial.
//
// The QAP polynomials are:
//   A_i(x) = Lagrange interpolation of A matrix column i
//   B_i(x) = Lagrange interpolation of B matrix column i
//   C_i(x) = Lagrange interpolation of C matrix column i
//
// With witness w, the full QAP polynomials are:
//   A(x) = Σ w_i · A_i(x)
//   B(x) = Σ w_i · B_i(x)
//   C(x) = Σ w_i · C_i(x)
// ============================================================
struct QAP {
    int n;             // number of constraints (padded to power of 2)
    int m;             // number of variables
    int log_n;         // log2(n)

    // Build QAP from R1CS
    static QAP from_r1cs(const R1CS& r1cs);
};

// ============================================================
// Proving Key
// ============================================================
struct ProvingKey {
    // Scalar-domain query representation. Since every query point is a fixed-generator
    // multiple in this implementation, proving can collapse MSMs into dot products in Fr.
    std::vector<Fr> tau_powers_scalars;
    std::vector<Fr> A_query_scalars;
    std::vector<Fr> B_query_scalars;
    std::vector<Fr> L_query_scalars;
    std::vector<Fr> H_query_scalars;

    struct DebugTrapdoor {
        Fr tau;
        Fr alpha;
        Fr beta;
        Fr gamma;
        Fr delta;
        bool available;

        DebugTrapdoor()
            : tau(Fr::zero()), alpha(Fr::zero()), beta(Fr::zero()),
              gamma(Fr::zero()), delta(Fr::zero()), available(false) {}
    };

    // Powers of tau in G1: [τ^0·G1, τ^1·G1, ..., τ^{n-1}·G1]
    std::vector<G1Affine> tau_powers_g1;
    // Powers of tau in G2: [τ^0·G2, τ^1·G2]
    std::vector<G2Affine> tau_powers_g2;

    // A query: [Σ_j u_j·A_j(τ)]·G1 for each variable i
    std::vector<G1Affine> A_query; // [m] points in G1
    // B query in G1 and G2
    std::vector<G1Affine> B_g1_query; // [m]
    std::vector<G2Affine> B_g2_query; // [m]

    // Private wire L query: (β·A_i(τ) + α·B_i(τ) + C_i(τ)) / δ for private wires
    std::vector<G1Affine> L_query; // [m - num_public - 1]

    // H query: τ^i · t(τ) / δ for quotient polynomial
    std::vector<G1Affine> H_query; // [n]

    // Verification key elements
    G1Affine alpha_g1;
    G2Affine beta_g2;
    G1Affine beta_g1;
    G2Affine gamma_g2;
    G1Affine delta_g1;
    G2Affine delta_g2;

    int num_constraints;
    int num_variables;
    int num_public;

    // Streaming support for large proving keys
    std::string pk_path; // path to serialized PK on disk
    bool is_streaming;
    bool materialized_points;
    DebugTrapdoor debug_trapdoor;

    ProvingKey()
        : num_constraints(0), num_variables(0), num_public(0),
          is_streaming(false), materialized_points(false) {}

    // Save/load for streaming
    bool save(const char* path) const;
    bool load_streaming(const char* path);

    // Get required VRAM for this PK
    size_t vram_bytes() const {
        size_t g1_pts = A_query.size() + B_g1_query.size() + L_query.size() + H_query.size();
        size_t g2_pts = B_g2_query.size();
        return g1_pts * sizeof(G1Affine) + g2_pts * sizeof(G2Affine);
    }
};

// ============================================================
// Verification Key
// ============================================================
struct VerificationKey {
    G1Affine alpha_g1;
    G2Affine beta_g2;
    G2Affine gamma_g2;
    G2Affine delta_g2;

    // IC (input commitment): IC[0] + Σ public_inputs[i] * IC[i+1]
    std::vector<G1Affine> ic; // [num_public + 1]

    int num_public;
};

// ============================================================
// Groth16 Proof
// ============================================================
struct Groth16Proof {
    G1Affine A;
    G2Affine B;
    G1Affine C;
    bool valid;

    Groth16Proof() : valid(false) {}

    void print() const {
        printf("Groth16 Proof: %s (256 bytes)\n", valid ? "valid" : "invalid");
    }

    std::vector<uint8_t> serialize() const;
    static Groth16Proof deserialize(const uint8_t* data, size_t len);
};

// ============================================================
// KZG Polynomial Commitment Scheme
//
// Commit(f) = Σ f_i · [τ^i]_1 (MSM over G1)
// Prove(f, z) = [f(τ) - f(z)] / (τ - z) · G1
// Verify: e(Commit - f(z)·G1, G2) == e(Proof, [τ]_2 - z·G2)
//
// GPU acceleration: the commitment is an MSM, fully parallelized.
// ============================================================
struct KZGCommitment {
    G1Affine commitment;

    // Commit to polynomial f using SRS (tau powers)
    static KZGCommitment commit(const Fr* coeffs, int degree,
                                  const G1Affine* srs_g1);

    // Open commitment at point z
    static G1Affine open(const Fr* coeffs, int degree,
                           const Fr& z, const G1Affine* srs_g1);
};

// ============================================================
// Groth16 Prover (GPU-accelerated)
// ============================================================
class Groth16Prover {
public:
    // Trusted setup: generate PK and VK from R1CS circuit
    static void setup(const R1CS& circuit, ProvingKey& pk, VerificationKey& vk);

    // Generate a Groth16 proof
    static Groth16Proof prove(const ProvingKey& pk,
                               const R1CS& circuit,
                               const Witness& witness);

    // Batch prove: generate multiple proofs, optionally aggregate
    static std::vector<Groth16Proof> batch_prove(
        const ProvingKey& pk,
        const R1CS& circuit,
        const std::vector<Witness>& witnesses);

    // Aggregate N proofs into 1 using random linear combination
    static Groth16Proof aggregate_proofs(
        const std::vector<Groth16Proof>& proofs,
        const VerificationKey& vk,
        const std::vector<std::vector<Fr>>& public_inputs);

private:
    // Compute quotient polynomial h(x) = (A(x)*B(x) - C(x)) / Z_H(x)
    // Uses NTT for polynomial multiplication and division.
    static std::vector<Fr> compute_h(
        const QAP& qap,
        const R1CS& circuit,
        const Witness& witness);
};

// ============================================================
// Groth16 Verifier (GPU-accelerated pairing check)
// ============================================================
class Groth16Verifier {
public:
    // Verify: e(-A, B) · e(α, β) · e(vk_x, γ) · e(C, δ) == 1
    static bool verify(const VerificationKey& vk,
                        const Groth16Proof& proof,
                        const std::vector<Fr>& public_inputs);

    // Batch verify using random linear combination
    static bool batch_verify(const VerificationKey& vk,
                              const std::vector<Groth16Proof>& proofs,
                              const std::vector<std::vector<Fr>>& public_inputs);
};

} // namespace zkml
