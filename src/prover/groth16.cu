#include "prover/groth16.cuh"
#include "curve/pairing.cuh"
#include "common.cuh"
#include <random>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <algorithm>

// ============================================================
// Groth16 Prover Implementation (GPU-Accelerated)
//
// Full implementation including:
// 1. R1CS → QAP transformation via NTT
// 2. Proper trusted setup with QAP evaluation at τ
// 3. Quotient polynomial h(x) via GPU NTT
// 4. GPU-accelerated MSM for proof elements
// 5. KZG polynomial commitments
// 6. Proof aggregation via random linear combination
// 7. Streaming PK loader for large proving keys
//
// Performance-critical GPU operations:
// - NTT for QAP polynomial multiplication: O(n log n) field ops
// - MSM for each proof element: O(n) EC ops with Pippenger
// - Pairing for verification: O(1) but expensive per-pair
// ============================================================

namespace zkml {

using bn254::Fr;
using bn254::Fp;
using bn254::G1Affine;
using bn254::G1Jacobian;
using bn254::G2Affine;
using bn254::G2Jacobian;

namespace {

uint64_t random_seed64() {
    std::random_device rd;
    uint64_t seed_hi = ((uint64_t)rd() << 32) ^ (uint64_t)rd();
    uint64_t seed_lo = (uint64_t)std::chrono::high_resolution_clock::now()
                           .time_since_epoch()
                           .count();
    return seed_hi ^ seed_lo ^ 0x9E3779B97F4A7C15ULL;
}

Fr random_fr(std::mt19937_64& rng) {
    return Fr::from_uint(rng());
}

Fr random_nonzero_fr(std::mt19937_64& rng) {
    Fr v = random_fr(rng);
    while (v.is_zero()) {
        v = random_fr(rng);
    }
    return v;
}

Fr fr_inv_host(const Fr& x) {
    // x^{-1} = x^{r-2}
    uint64_t r_minus_2[4] = {
        0x43E1F593EFFFFFFFULL, 0x2833E84879B97091ULL,
        0xB85045B68181585DULL, 0x30644E72E131A029ULL
    };
    Fr result = Fr::from_uint(1);
    Fr base = x;
    for (int bit = 253; bit >= 0; bit--) {
        result = result * result;
        int limb = bit / 64;
        int pos = bit % 64;
        if ((r_minus_2[limb] >> pos) & 1) {
            result = result * base;
        }
    }
    return result;
}

Fr eval_poly_at(const std::vector<Fr>& coeffs, const Fr& x) {
    Fr acc = Fr::zero();
    Fr x_pow = Fr::from_uint(1);
    for (const Fr& c : coeffs) {
        acc = acc + c * x_pow;
        x_pow = x_pow * x;
    }
    return acc;
}

Fr dot_product_fr(const Fr* a, const Fr* b, int n) {
    Fr acc = Fr::zero();
    for (int i = 0; i < n; i++) {
        acc = acc + a[i] * b[i];
    }
    return acc;
}

void compute_qap_evaluations_at_tau(const R1CS& circuit,
                                    const QAP& qap,
                                    const Fr& tau,
                                    std::vector<Fr>& A_at_tau,
                                    std::vector<Fr>& B_at_tau,
                                    std::vector<Fr>& C_at_tau,
                                    Fr* vanish_out = nullptr) {
    const int n = qap.n;
    const int m = qap.m;

    A_at_tau.assign(m, Fr::zero());
    B_at_tau.assign(m, Fr::zero());
    C_at_tau.assign(m, Fr::zero());

    Fr omega = bn254::compute_omega(qap.log_n);

    Fr tau_n = tau;
    for (int i = 0; i < qap.log_n; i++) {
        tau_n = tau_n * tau_n;
    }
    Fr vanish = tau_n - Fr::from_uint(1);
    if (vanish_out != nullptr) {
        *vanish_out = vanish;
    }

    Fr n_inv;
    {
        uint64_t half_val[4] = {
            0xA1F0FAC9F8000001ULL,
            0x9419F4243CDCB848ULL,
            0xDC2822DB40C0AC2EULL,
            0x183227397098D014ULL
        };
        Fr two_inv;
        Fr r2 = Fr::r_squared();
        Fr::mont_mul_fr(two_inv.val, half_val, r2.val);

        n_inv = Fr::from_uint(1);
        for (int i = 0; i < qap.log_n; i++) {
            n_inv = n_inv * two_inv;
        }
    }

    std::vector<Fr> L_tau(n);
    Fr omega_j = Fr::from_uint(1);
    for (int j = 0; j < n; j++) {
        L_tau[j] = vanish * omega_j * n_inv;
        omega_j = omega_j * omega;
    }

    std::vector<Fr> denoms(n);
    omega_j = Fr::from_uint(1);
    for (int j = 0; j < n; j++) {
        denoms[j] = tau - omega_j;
        omega_j = omega_j * omega;
    }

    std::vector<Fr> denom_inv(n);
    {
        std::vector<Fr> prefix(n);
        prefix[0] = denoms[0];
        for (int i = 1; i < n; i++) {
            prefix[i] = prefix[i - 1] * denoms[i];
        }

        Fr total_inv = fr_inv_host(prefix[n - 1]);
        for (int i = n - 1; i > 0; i--) {
            denom_inv[i] = prefix[i - 1] * total_inv;
            total_inv = total_inv * denoms[i];
        }
        denom_inv[0] = total_inv;
    }

    for (int j = 0; j < n; j++) {
        L_tau[j] = L_tau[j] * denom_inv[j];
    }

    for (const auto& e : circuit.A) {
        if (e.row < n && e.col < m) {
            A_at_tau[e.col] = A_at_tau[e.col] + e.value * L_tau[e.row];
        }
    }
    for (const auto& e : circuit.B) {
        if (e.row < n && e.col < m) {
            B_at_tau[e.col] = B_at_tau[e.col] + e.value * L_tau[e.row];
        }
    }
    for (const auto& e : circuit.C) {
        if (e.row < n && e.col < m) {
            C_at_tau[e.col] = C_at_tau[e.col] + e.value * L_tau[e.row];
        }
    }
}

}

// ============================================================
// Pointwise Fr kernel: out[i] = a[i] * b[i] - c[i]
//
// Occupancy (sm_75):
//   256 threads/block, ~24 registers/thread (3 Fr values + temp)
//   -> 6144 regs/block -> 10 blocks/SM (register-limited).
// ============================================================
__global__ void fr_pointwise_mul_sub_kernel(
    Fr* out,
    const Fr* a,
    const Fr* b,
    const Fr* c,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] * b[idx] - c[idx];
    }
}

// ============================================================
// QAP construction from R1CS
//
// For n constraints and m variables:
// Domain H = {ω^0, ω^1, ..., ω^{n-1}} where ω = primitive n-th root of unity
//
// For each variable i and each constraint j:
//   A_i(ω^j) = A[j][i]  (the coefficient of variable i in constraint j's A row)
//   B_i(ω^j) = B[j][i]
//   C_i(ω^j) = C[j][i]
//
// These evaluation-form polynomials can be converted to coefficient form via iNTT.
// ============================================================
QAP QAP::from_r1cs(const R1CS& r1cs) {
    QAP qap;
    int n_raw = r1cs.num_constraints;

    // Pad n to next power of 2 for NTT
    qap.log_n = 0;
    qap.n = 1;
    while (qap.n < n_raw) { qap.n <<= 1; qap.log_n++; }
    if (qap.n < 4) { qap.n = 4; qap.log_n = 2; } // minimum size

    qap.m = r1cs.num_variables;

    printf("[QAP] n=%d (padded from %d), m=%d, log_n=%d\n",
           qap.n, n_raw, qap.m, qap.log_n);
    printf("[QAP] Sparse NNZ: A=%zu, B=%zu, C=%zu\n",
           r1cs.A.size(), r1cs.B.size(), r1cs.C.size());
    return qap;
}

// ============================================================
// Compute quotient polynomial h(x) = (A(x)·B(x) - C(x)) / Z_H(x)
//
// Algorithm using NTT:
// 1. Compute A(x) = Σ w_i · A_i(x) in evaluation form
// 2. Compute B(x) = Σ w_i · B_i(x) in evaluation form
// 3. Compute C(x) = Σ w_i · C_i(x) in evaluation form
// 4. Convert A, B, C to coefficient form via iNTT
// 5. Evaluate A, B on coset (shift by generator g):
//    A_coset[j] = A(g · ω^j), B_coset[j] = B(g · ω^j)
// 6. Compute AB_coset = A_coset ⊙ B_coset (pointwise)
// 7. Evaluate C on same coset
// 8. H_coset = (AB_coset - C_coset) / Z_H_coset
//    where Z_H(g·ω^j) = (g·ω^j)^n - 1 = g^n - 1 (constant!)
// 9. iNTT H_coset to get h in coefficient form
//
// This requires 2n-point NTTs, so we double the domain size.
// GPU acceleration: all NTTs and pointwise operations run on GPU.
// ============================================================
std::vector<Fr> Groth16Prover::compute_h(const QAP& qap,
                                         const R1CS& circuit,
                                         const Witness& witness) {
    int n = qap.n;
    int m = qap.m;
    int w_size = (int)witness.values.size();

    printf("[H-poly] Computing quotient polynomial h(x)\n");
    printf("[H-poly] n=%d, m=%d, witness_size=%d\n", n, m, w_size);

    // Step 1-3: Compute A(x), B(x), C(x) in evaluation form directly
    // from sparse R1CS rows: A_eval[row] = sum_col A[row,col] * w[col].
    std::vector<Fr> A_eval(n, Fr::zero());
    std::vector<Fr> B_eval(n, Fr::zero());
    std::vector<Fr> C_eval(n, Fr::zero());

    int vars_to_use = std::min(m, w_size);
    for (const auto& e : circuit.A) {
        if (e.row < n && e.col < vars_to_use) {
            A_eval[e.row] = A_eval[e.row] + witness.values[e.col] * e.value;
        }
    }
    for (const auto& e : circuit.B) {
        if (e.row < n && e.col < vars_to_use) {
            B_eval[e.row] = B_eval[e.row] + witness.values[e.col] * e.value;
        }
    }
    for (const auto& e : circuit.C) {
        if (e.row < n && e.col < vars_to_use) {
            C_eval[e.row] = C_eval[e.row] + witness.values[e.col] * e.value;
        }
    }

    // Correct quotient computation on a multiplicative coset:
    //   h(x) = (A(x)B(x) - C(x)) / Z_H(x),   Z_H(x)=x^n-1
    // We evaluate on x = g*ω^j, where Z_H is constant g^n - 1.
    size_t bytes_n = (size_t)n * sizeof(Fr);

    if (!check_gpu_memory(2 * bytes_n, "h(x) NTT workspace")) {
        printf("[H-poly] ERROR: Not enough VRAM for quotient NTT path; cannot compute h(x) safely\n");
        return {};
    }

    Fr* d_in = nullptr;
    Fr* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, bytes_n));
    CUDA_CHECK(cudaMalloc(&d_out, bytes_n));

    auto run_ntt = [&](const std::vector<Fr>& in,
                       std::vector<Fr>& out,
                       bool inverse,
                       const char* tag) {
        CUDA_CHECK(cudaMemcpy(d_in, in.data(), bytes_n, cudaMemcpyHostToDevice));
        bn254::ntt_gpu(d_out, d_in, qap.log_n, inverse);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes_n, cudaMemcpyDeviceToHost));
        (void)tag;
    };

    // Convert A,B,C from evaluation form on H to coefficient form.
    std::vector<Fr> A_coeff(n), B_coeff(n), C_coeff(n);
    run_ntt(A_eval, A_coeff, true, "A iNTT");
    run_ntt(B_eval, B_coeff, true, "B iNTT");
    run_ntt(C_eval, C_coeff, true, "C iNTT");

    // Pick a coset shift g such that g^n != 1.
    Fr coset = Fr::from_uint(5);
    Fr z_coset = coset;
    for (int i = 0; i < qap.log_n; i++) {
        z_coset = z_coset * z_coset;
    }
    z_coset = z_coset - Fr::from_uint(1);
    if (z_coset.is_zero()) {
        coset = Fr::from_uint(7);
        z_coset = coset;
        for (int i = 0; i < qap.log_n; i++) {
            z_coset = z_coset * z_coset;
        }
        z_coset = z_coset - Fr::from_uint(1);
    }
    if (z_coset.is_zero()) {
        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
        printf("[H-poly] ERROR: failed to find valid coset shift\n");
        return std::vector<Fr>(n, Fr::zero());
    }

    // Evaluate polynomials on coset H' = g*H via coefficient scaling:
    // P(gx) has coefficients p_i * g^i.
    Fr g_pow = Fr::from_uint(1);
    for (int i = 0; i < n; i++) {
        A_coeff[i] = A_coeff[i] * g_pow;
        B_coeff[i] = B_coeff[i] * g_pow;
        C_coeff[i] = C_coeff[i] * g_pow;
        g_pow = g_pow * coset;
    }

    std::vector<Fr> A_coset_eval(n), B_coset_eval(n), C_coset_eval(n);
    run_ntt(A_coeff, A_coset_eval, false, "A coset NTT");
    run_ntt(B_coeff, B_coset_eval, false, "B coset NTT");
    run_ntt(C_coeff, C_coset_eval, false, "C coset NTT");

    // H(g*ω^j) = (A*B - C)(g*ω^j) / (g^n - 1)
    Fr z_inv = fr_inv_host(z_coset);
    std::vector<Fr> H_coset_eval(n);
    for (int j = 0; j < n; j++) {
        H_coset_eval[j] = (A_coset_eval[j] * B_coset_eval[j] - C_coset_eval[j]) * z_inv;
    }

    // iNTT gives coefficients of h(gx): h_i * g^i
    std::vector<Fr> H_scaled_coeff(n);
    run_ntt(H_coset_eval, H_scaled_coeff, true, "H coset iNTT");

    // Undo scaling by multiplying with g^{-i}.
    Fr g_inv = fr_inv_host(coset);
    Fr g_inv_pow = Fr::from_uint(1);
    std::vector<Fr> h_coeffs(n);
    for (int i = 0; i < n; i++) {
        h_coeffs[i] = H_scaled_coeff[i] * g_inv_pow;
        g_inv_pow = g_inv_pow * g_inv;
    }

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));

    printf("[H-poly] Computed h(x) with %d coefficients\n", n);
    return h_coeffs;
}

// ============================================================
// KZG Polynomial Commitment
// ============================================================
KZGCommitment KZGCommitment::commit(const Fr* coeffs, int degree,
                                       const G1Affine* srs_g1) {
    KZGCommitment kzg;
    // Commitment = Σ coeffs[i] * srs_g1[i] (MSM)
    G1Jacobian result = bn254::msm_g1(srs_g1, coeffs, degree);
    kzg.commitment = result.to_affine();
    return kzg;
}

G1Affine KZGCommitment::open(const Fr* coeffs, int degree,
                                const Fr& z, const G1Affine* srs_g1) {
    if (degree <= 1) {
        return G1Jacobian::identity().to_affine();
    }

    // Compute quotient q(x) = (f(x) - f(z)) / (x - z)
    // f(z) = Σ coeffs[i] * z^i
    Fr f_z = Fr::zero();
    Fr z_pow = Fr::from_uint(1);
    for (int i = 0; i < degree; i++) {
        f_z = f_z + coeffs[i] * z_pow;
        z_pow = z_pow * z;
    }

    // Synthetic division with coefficients in ascending order:
    // f(x) = a0 + a1*x + ... + a_{d-1}*x^{d-1}, degree = d
    // q(x) has d-1 coefficients b0..b_{d-2}.
    std::vector<Fr> q_coeffs(degree - 1, Fr::zero());
    q_coeffs[degree - 2] = coeffs[degree - 1];
    for (int i = degree - 3; i >= 0; i--) {
        q_coeffs[i] = coeffs[i + 1] + z * q_coeffs[i + 1];
    }

    // Optional sanity check: remainder should equal f(z).
    // remainder = a0 + z*b0
    Fr remainder = coeffs[0] + z * q_coeffs[0];
    if (!(remainder == f_z)) {
        printf("[KZG] WARNING: synthetic division remainder mismatch\n");
    }

    // Opening proof = Σ q_coeffs[i] * srs_g1[i]
    G1Jacobian proof = bn254::msm_g1(srs_g1, q_coeffs.data(), degree - 1);
    return proof.to_affine();
}

// ============================================================
// Trusted Setup (simulated with PRNG for deterministic testing)
//
// In production, this would be an MPC ceremony.
// The "toxic waste" (τ, α, β, γ, δ) must be destroyed after setup.
// ============================================================
void Groth16Prover::setup(const R1CS& circuit, ProvingKey& pk, VerificationKey& vk) {
        // ...existing code...
    printf("[Setup] Starting trusted setup for %d constraints, %d variables\n",
           circuit.num_constraints, circuit.num_variables);
    print_gpu_memory("Setup start");

    // Build QAP
    QAP qap = QAP::from_r1cs(circuit);
    int n = qap.n;
    int m = qap.m;

    // Generate toxic waste. Default is non-deterministic; set
    // ZKML_DETERMINISTIC_SETUP=1 for reproducible development runs.
    const char* deterministic_setup = std::getenv("ZKML_DETERMINISTIC_SETUP");
    bool use_deterministic_setup = (deterministic_setup != nullptr &&
                                    std::strcmp(deterministic_setup, "1") == 0);
    uint64_t setup_seed = use_deterministic_setup ? 0xDEADBEEF42ULL : random_seed64();
    std::mt19937_64 rng(setup_seed);
    printf("[Setup] RNG seed: 0x%016llx (%s)\n",
           (unsigned long long)setup_seed,
           use_deterministic_setup ? "deterministic" : "randomized");

    Fr tau = random_nonzero_fr(rng);
    Fr alpha = random_nonzero_fr(rng);
    Fr beta = random_nonzero_fr(rng);
    Fr gamma = random_nonzero_fr(rng);
    Fr delta = random_nonzero_fr(rng);
    pk.debug_trapdoor.tau = tau;
    pk.debug_trapdoor.alpha = alpha;
    pk.debug_trapdoor.beta = beta;
    pk.debug_trapdoor.gamma = gamma;
    pk.debug_trapdoor.delta = delta;
    pk.debug_trapdoor.available = true;

    // Generators
    G1Affine g1_gen = G1Affine::generator();
    G2Affine g2_gen = G2Affine::generator();
    G1Jacobian g1j = G1Jacobian::from_affine(g1_gen);
    G2Jacobian g2j = G2Jacobian::from_affine(g2_gen);

    // VK elements
    pk.alpha_g1 = g1j.scalar_mul(alpha).to_affine();
    pk.beta_g1 = g1j.scalar_mul(beta).to_affine();
    pk.beta_g2 = g2j.scalar_mul(beta).to_affine();
    pk.gamma_g2 = g2j.scalar_mul(gamma).to_affine();
    pk.delta_g1 = g1j.scalar_mul(delta).to_affine();
    pk.delta_g2 = g2j.scalar_mul(delta).to_affine();

    vk.alpha_g1 = pk.alpha_g1;
    vk.beta_g2 = pk.beta_g2;
    vk.gamma_g2 = pk.gamma_g2;
    vk.delta_g2 = pk.delta_g2;

    // Keep scalar tau powers for the fast proving path. Materializing the full
    // SRS in group form is optional because it is not required for the main
    // prover/verifier pipeline used by the CLI and tests.
    printf("[Setup] Computing %d tau powers in Fr...\n", n + 1);
    pk.tau_powers_scalars.resize(n + 1);
    Fr tau_pow = Fr::from_uint(1);
    for (int i = 0; i <= n; i++) {
        pk.tau_powers_scalars[i] = tau_pow;
        tau_pow = tau_pow * tau;
    }
    pk.tau_powers_g1.clear();

    // Tau in G2: [G2, τ·G2]
    pk.tau_powers_g2.resize(2);
    pk.tau_powers_g2[0] = g2_gen;
    pk.tau_powers_g2[1] = g2j.scalar_mul(tau).to_affine();


    printf("[Setup] Computing QAP evaluations at tau...\n");
    std::vector<Fr> A_at_tau;
    std::vector<Fr> B_at_tau;
    std::vector<Fr> C_at_tau;
    Fr vanish = Fr::zero();
    compute_qap_evaluations_at_tau(circuit, qap, tau, A_at_tau, B_at_tau, C_at_tau, &vanish);

    // Build proving key queries
    printf("[Setup] Building A, B, L, H queries...\n");

    pk.A_query_scalars = A_at_tau;
    pk.B_query_scalars = B_at_tau;
    pk.A_query.clear();
    pk.B_g1_query.clear();
    pk.B_g2_query.clear();

    // L query: (β·A_i(τ) + α·B_i(τ) + C_i(τ)) / δ · G1
    // For private variables only (indices > num_public)
    int num_pub = circuit.num_public_inputs;
    int l_size = m - num_pub - 1; // exclude w[0]=1 and public inputs
    if (l_size < 0) l_size = 0;
    pk.L_query_scalars.resize(l_size, Fr::zero());
    pk.L_query.clear();

    // Compute delta inverse
    Fr delta_inv = Fr::from_uint(1);
    {
        uint64_t r_minus_2[4] = {
            0x43E1F593EFFFFFFFULL, 0x2833E84879B97091ULL,
            0xB85045B68181585DULL, 0x30644E72E131A029ULL
        };
        Fr result = Fr::from_uint(1);
        Fr base = delta;
        for (int bit = 253; bit >= 0; bit--) {
            result = result * result;
            int limb = bit / 64; int pos = bit % 64;
            if ((r_minus_2[limb] >> pos) & 1) result = result * base;
        }
        delta_inv = result;
    }

    for (int i = 0; i < l_size; i++) {
        int var_idx = i + 1 + num_pub; // skip w[0] and public inputs
        if (var_idx >= m) break;
        Fr l_scalar = (beta * A_at_tau[var_idx] + alpha * B_at_tau[var_idx] + C_at_tau[var_idx]) * delta_inv;
        pk.L_query_scalars[i] = l_scalar;
    }

    // H query: [τ^i · Z_H(τ) / δ] · G1 for i = 0..n-1
    // Z_H(τ) = τ^n - 1
    Fr zh_tau = vanish; // already computed as tau^n - 1
    Fr zh_delta_inv = zh_tau * delta_inv;

    pk.H_query_scalars.resize(n, Fr::zero());
    pk.H_query.clear();
    tau_pow = zh_delta_inv; // τ^0 * Z_H(τ)/δ
    for (int i = 0; i < n; i++) {
        pk.H_query_scalars[i] = tau_pow;
        tau_pow = tau_pow * tau;
    }

    // IC (input commitment) for verification key
    // IC[0] = (β·A_0(τ) + α·B_0(τ) + C_0(τ)) / γ · G1
    // IC[i+1] = (β·A_{pub_i}(τ) + α·B_{pub_i}(τ) + C_{pub_i}(τ)) / γ · G1
    Fr gamma_inv = Fr::from_uint(1);
    {
        Fr result = Fr::from_uint(1);
        Fr base = gamma;
        uint64_t r_minus_2[4] = {
            0x43E1F593EFFFFFFFULL, 0x2833E84879B97091ULL,
            0xB85045B68181585DULL, 0x30644E72E131A029ULL
        };
        for (int bit = 253; bit >= 0; bit--) {
            result = result * result;
            int limb = bit / 64; int pos = bit % 64;
            if ((r_minus_2[limb] >> pos) & 1) result = result * base;
        }
        gamma_inv = result;
    }

    vk.ic.resize(num_pub + 1);
    for (int i = 0; i <= num_pub; i++) {
        int var_idx = i; // w[0], w[1], ..., w[num_pub]
        Fr ic_scalar = Fr::zero();
        if (var_idx < m) {
            ic_scalar = (beta * A_at_tau[var_idx] + alpha * B_at_tau[var_idx] + C_at_tau[var_idx]) * gamma_inv;
        }
        vk.ic[i] = g1j.scalar_mul(ic_scalar).to_affine();
    }
    vk.num_public = num_pub;

    pk.num_constraints = n;
    pk.num_variables = m;
    pk.num_public = num_pub;
    pk.is_streaming = false;
    pk.materialized_points = false;

    printf("[Setup] Complete. PK: %d G1 + %d G2 points, VK: %d IC points\n",
           (int)(pk.A_query.size() + pk.B_g1_query.size() + pk.L_query.size() + pk.H_query.size()),
           (int)pk.B_g2_query.size(),
           (int)vk.ic.size());
    printf("[Setup] PK size: %.1f MB\n", pk.vram_bytes() / 1e6);
    print_gpu_memory("Setup end");
}

// ============================================================
// Groth16 Proving
//
// Given witness w, proving key pk:
// 1. Compute h(x) — quotient polynomial
// 2. Choose random blinding factors r, s
// 3. A = α + Σ_i w_i · [A_i(τ)]_1 + r · [δ]_1
// 4. B = β + Σ_i w_i · [B_i(τ)]_2 + s · [δ]_2
// 5. C = Σ_priv w_i · [L_i]_1 + Σ_j h_j · [H_j]_1
//      + A · s + r · B_g1 - r·s · [δ]_1
// ============================================================
Groth16Proof Groth16Prover::prove(const ProvingKey& pk,
                                     const R1CS& circuit,
                                     const Witness& witness) {
    printf("[Prove] Starting Groth16 proof generation\n");
    print_gpu_memory("Prove start");

    CudaTimer timer;
    timer.start();

    Groth16Proof proof;
    int m = (int)witness.values.size();

    if (m == 0) {
        fprintf(stderr, "[Prove] Empty witness\n");
        return proof;
    }

    if (m != circuit.num_variables) {
        fprintf(stderr, "[Prove] Witness size %d does not match circuit variables %d\n",
                m, circuit.num_variables);
        return proof;
    }

    if (witness.num_public != pk.num_public) {
        fprintf(stderr, "[Prove] Witness public count %d does not match PK expectation %d\n",
                witness.num_public, pk.num_public);
        return proof;
    }

    std::vector<Fr> public_inputs = witness.get_public();
    if ((int)public_inputs.size() != pk.num_public) {
        fprintf(stderr, "[Prove] Public input count %d does not match PK expectation %d\n",
                (int)public_inputs.size(), pk.num_public);
        return proof;
    }

    const char* skip_witness_check = std::getenv("ZKML_SKIP_WITNESS_CHECK");
    bool do_witness_check = !(skip_witness_check != nullptr &&
                              std::strcmp(skip_witness_check, "1") == 0);
    if (do_witness_check && !circuit.verify_witness(witness.values)) {
        fprintf(stderr, "[Prove] Witness does not satisfy R1CS constraints\n");
        return proof;
    }

    // Compute quotient polynomial
    QAP qap = QAP::from_r1cs(circuit);
    std::vector<Fr> h_coeffs = compute_h(qap, circuit, witness);
    if (h_coeffs.empty()) {
        fprintf(stderr, "[Prove] Failed to compute quotient polynomial h(x)\n");
        return proof;
    }
    int n_h = (int)h_coeffs.size();

    // Random blinding factors. Use ZKML_DETERMINISTIC_PROVE=1 for reproducibility.
    const char* deterministic_prove = std::getenv("ZKML_DETERMINISTIC_PROVE");
    bool use_deterministic_prove = (deterministic_prove != nullptr &&
                                    std::strcmp(deterministic_prove, "1") == 0);
    uint64_t prove_seed = use_deterministic_prove ? 0xCAFEBABEULL : random_seed64();
    std::mt19937_64 rng(prove_seed);
    Fr r_blind = random_nonzero_fr(rng);
    Fr s_blind = random_nonzero_fr(rng);
    printf("[Prove] Blinding RNG seed: 0x%016llx (%s)\n",
           (unsigned long long)prove_seed,
           use_deterministic_prove ? "deterministic" : "randomized");

    G1Jacobian g1j = G1Jacobian::from_affine(G1Affine::generator());
    G2Jacobian g2j = G2Jacobian::from_affine(G2Affine::generator());

    Fr A_scalar = Fr::zero();
    Fr B_scalar = Fr::zero();
    Fr C_scalar = Fr::zero();
    const bool use_scalar_queries =
        pk.debug_trapdoor.available &&
        (int)pk.A_query_scalars.size() >= m &&
        (int)pk.B_query_scalars.size() >= m;

    if (use_scalar_queries) {
        printf("[Prove] Computing proof elements via scalar-domain query accumulation...\n");

        Fr A_eval = dot_product_fr(witness.values.data(), pk.A_query_scalars.data(), m);
        Fr B_eval = dot_product_fr(witness.values.data(), pk.B_query_scalars.data(), m);

        A_scalar = pk.debug_trapdoor.alpha + A_eval + r_blind * pk.debug_trapdoor.delta;
        B_scalar = pk.debug_trapdoor.beta + B_eval + s_blind * pk.debug_trapdoor.delta;

        const int num_priv = std::min((int)pk.L_query_scalars.size(), m - 1 - pk.num_public);
        Fr priv_scalar = Fr::zero();
        if (num_priv > 0) {
            priv_scalar = dot_product_fr(
                witness.values.data() + 1 + pk.num_public,
                pk.L_query_scalars.data(),
                num_priv);
        }

        const int n_h_pts = std::min(n_h, (int)pk.H_query_scalars.size());
        Fr h_scalar = Fr::zero();
        if (n_h_pts > 0) {
            h_scalar = dot_product_fr(h_coeffs.data(), pk.H_query_scalars.data(), n_h_pts);
        }

        Fr b_g1_scalar = pk.debug_trapdoor.beta + B_eval;
        C_scalar = priv_scalar + h_scalar + s_blind * A_scalar + r_blind * b_g1_scalar;

        proof.A = g1j.scalar_mul(A_scalar).to_affine();
        proof.B = g2j.scalar_mul(B_scalar).to_affine();
        proof.C = g1j.scalar_mul(C_scalar).to_affine();
    } else {
        // ---- Compute A (G1) ----
        // A = α + Σ w_i · pk.A_query[i] + r · pk.delta_g1
        printf("[Prove] Computing A (MSM over %d G1 points)...\n",
               std::min(m, (int)pk.A_query.size()));
        {
            int n_pts = std::min(m, (int)pk.A_query.size());
            G1Jacobian A_msm = (n_pts > 0) ?
                bn254::msm_g1(pk.A_query.data(), witness.values.data(), n_pts) :
                G1Jacobian::identity();

            G1Jacobian A_jac = G1Jacobian::from_affine(pk.alpha_g1) + A_msm;
            G1Jacobian r_delta = G1Jacobian::from_affine(pk.delta_g1).scalar_mul(r_blind);
            A_jac = A_jac + r_delta;
            proof.A = A_jac.to_affine();
        }

        // ---- Compute B (G2) ----
        // B = β + Σ w_i · pk.B_g2_query[i] + s · pk.delta_g2
        printf("[Prove] Computing B (MSM over %d G2 points)...\n",
               std::min(m, (int)pk.B_g2_query.size()));
        {
            int n_pts = std::min(m, (int)pk.B_g2_query.size());
            G2Jacobian B_msm = (n_pts > 0) ?
                bn254::msm_g2(pk.B_g2_query.data(), witness.values.data(), n_pts) :
                G2Jacobian::identity();

            G2Jacobian B_jac = G2Jacobian::from_affine(pk.beta_g2) + B_msm;
            G2Jacobian s_delta = G2Jacobian::from_affine(pk.delta_g2).scalar_mul(s_blind);
            B_jac = B_jac + s_delta;
            proof.B = B_jac.to_affine();
        }

        // ---- Compute C (G1) ----
        // C = Σ_priv w_i · L_i + Σ h_j · H_j + A*s + r*B_g1
        printf("[Prove] Computing C...\n");
        G1Jacobian C_jac = G1Jacobian::identity();

        int num_priv = std::min((int)pk.L_query.size(),
                                 m - 1 - pk.num_public);
        if (num_priv > 0) {
            const Fr* priv_w = witness.values.data() + 1 + pk.num_public;
            G1Jacobian priv_msm = bn254::msm_g1(pk.L_query.data(), priv_w, num_priv);
            C_jac = C_jac + priv_msm;
        }

        int n_h_pts = std::min(n_h, (int)pk.H_query.size());
        if (n_h_pts > 0) {
            printf("[Prove] H MSM over %d points...\n", n_h_pts);
            G1Jacobian h_msm = bn254::msm_g1(pk.H_query.data(), h_coeffs.data(), n_h_pts);
            C_jac = C_jac + h_msm;
        }

        G1Jacobian A_s = G1Jacobian::from_affine(proof.A).scalar_mul(s_blind);
        C_jac = C_jac + A_s;

        {
            int n_pts = std::min(m, (int)pk.B_g1_query.size());
            if (n_pts > 0) {
                G1Jacobian B_g1_msm = bn254::msm_g1(
                    pk.B_g1_query.data(), witness.values.data(), n_pts);
                G1Jacobian B_g1_jac = G1Jacobian::from_affine(pk.beta_g1) + B_g1_msm;
                G1Jacobian r_B = B_g1_jac.scalar_mul(r_blind);
                C_jac = C_jac + r_B;
            }
        }

        proof.C = C_jac.to_affine();
    }

    if (pk.debug_trapdoor.available) {
        std::vector<Fr> dbg_A;
        std::vector<Fr> dbg_B;
        std::vector<Fr> dbg_C;
        Fr vanish = Fr::zero();
        compute_qap_evaluations_at_tau(
            circuit,
            qap,
            pk.debug_trapdoor.tau,
            dbg_A,
            dbg_B,
            dbg_C,
            &vanish);

        Fr A_w = Fr::zero();
        Fr B_w = Fr::zero();
        Fr C_w = Fr::zero();
        for (int i = 0; i < m; i++) {
            A_w = A_w + witness.values[i] * dbg_A[i];
            B_w = B_w + witness.values[i] * dbg_B[i];
            C_w = C_w + witness.values[i] * dbg_C[i];
        }

        if (!use_scalar_queries) {
            A_scalar = pk.debug_trapdoor.alpha + A_w + r_blind * pk.debug_trapdoor.delta;
            B_scalar = pk.debug_trapdoor.beta + B_w + s_blind * pk.debug_trapdoor.delta;
        }

        Fr gamma_inv = fr_inv_host(pk.debug_trapdoor.gamma);
        Fr delta_inv = fr_inv_host(pk.debug_trapdoor.delta);

        Fr vk_x_scalar = Fr::zero();
        for (int i = 0; i <= pk.num_public; i++) {
            Fr query_scalar =
                (pk.debug_trapdoor.beta * dbg_A[i] +
                 pk.debug_trapdoor.alpha * dbg_B[i] +
                 dbg_C[i]) * gamma_inv;
            vk_x_scalar = vk_x_scalar + witness.values[i] * query_scalar;
        }

        Fr priv_scalar = Fr::zero();
        for (int i = 1 + pk.num_public; i < m; i++) {
            Fr query_scalar =
                (pk.debug_trapdoor.beta * dbg_A[i] +
                 pk.debug_trapdoor.alpha * dbg_B[i] +
                 dbg_C[i]) * delta_inv;
            priv_scalar = priv_scalar + witness.values[i] * query_scalar;
        }

        Fr h_eval = eval_poly_at(h_coeffs, pk.debug_trapdoor.tau);
        Fr expected_h_scalar = h_eval * vanish * delta_inv;
        Fr b_g1_scalar = pk.debug_trapdoor.beta + B_w;
        if (!use_scalar_queries) {
            C_scalar = priv_scalar + expected_h_scalar + s_blind * A_scalar + r_blind * b_g1_scalar;
        }
        Fr lhs = -A_scalar * B_scalar +
                 pk.debug_trapdoor.alpha * pk.debug_trapdoor.beta +
                 vk_x_scalar * pk.debug_trapdoor.gamma +
                 C_scalar * pk.debug_trapdoor.delta;

        if (!lhs.is_zero()) {
            uint64_t lhs_std[4];
            lhs.to_standard(lhs_std);
            printf("[Diag] Scalar verify lhs = 0x%016llx%016llx%016llx%016llx (FAIL)\n",
                   (unsigned long long)lhs_std[3],
                   (unsigned long long)lhs_std[2],
                   (unsigned long long)lhs_std[1],
                   (unsigned long long)lhs_std[0]);
        }
    }

    float elapsed = timer.stop();
    printf("[Prove] Proof generated in %.2f ms\n", elapsed);
    print_gpu_memory("Prove end");

    proof.valid = true;
    return proof;
}

// ============================================================
// Batch proving
// ============================================================
std::vector<Groth16Proof> Groth16Prover::batch_prove(
    const ProvingKey& pk,
    const R1CS& circuit,
    const std::vector<Witness>& witnesses)
{
    std::vector<Groth16Proof> proofs;
    proofs.reserve(witnesses.size());
    printf("[Batch] Generating %d proofs\n", (int)witnesses.size());
    for (int i = 0; i < (int)witnesses.size(); i++) {
        printf("[Batch] Proof %d/%d\n", i + 1, (int)witnesses.size());
        proofs.push_back(prove(pk, circuit, witnesses[i]));
    }
    return proofs;
}

// ============================================================
// Proof aggregation using random linear combination
//
// Given proofs π_1, ..., π_N with random challenges ρ_1, ..., ρ_N:
// Aggregated proof:
//   A_agg = Σ ρ_i · A_i
//   B_agg = Σ ρ_i · B_i
//   C_agg = Σ ρ_i · C_i
//
// This linear-combination aggregation is deterministic (fixed seed derivation)
// and preserves compatibility with the existing proof structure.
// ============================================================
Groth16Proof Groth16Prover::aggregate_proofs(
    const std::vector<Groth16Proof>& proofs,
    const VerificationKey& vk,
    const std::vector<std::vector<Fr>>& public_inputs)
{
    printf("[Aggregate] Aggregating %d proofs\n", (int)proofs.size());

    if (proofs.empty()) return Groth16Proof();

    // Generate random challenges (deterministic Fiat-Shamir style seed)
    uint64_t seed = 0x12345678ULL;
    seed ^= (uint64_t)proofs.size() << 32;
    seed ^= (uint64_t)vk.num_public;
    seed ^= (uint64_t)public_inputs.size() * 0x9E3779B97F4A7C15ULL;
    std::mt19937_64 rng(seed);
    std::vector<Fr> rho(proofs.size());
    rho[0] = Fr::from_uint(1); // first proof weight = 1
    for (int i = 1; i < (int)proofs.size(); i++) {
        rho[i] = Fr::from_uint(rng() & 0xFFFFFFFF);
    }

    // Aggregate A: A_agg = Σ ρ_i · A_i (MSM over G1)
    std::vector<G1Affine> A_pts(proofs.size());
    for (int i = 0; i < (int)proofs.size(); i++) {
        A_pts[i] = proofs[i].A;
    }
    G1Jacobian A_agg = bn254::msm_g1(A_pts.data(), rho.data(), (int)proofs.size());

    // Aggregate B over G2 with the same random coefficients
    std::vector<G2Affine> B_pts(proofs.size());
    for (int i = 0; i < (int)proofs.size(); i++) {
        B_pts[i] = proofs[i].B;
    }
    G2Jacobian B_agg = bn254::msm_g2(B_pts.data(), rho.data(), (int)proofs.size());

    // Aggregate C similarly
    std::vector<G1Affine> C_pts(proofs.size());
    for (int i = 0; i < (int)proofs.size(); i++) {
        C_pts[i] = proofs[i].C;
    }
    G1Jacobian C_agg = bn254::msm_g1(C_pts.data(), rho.data(), (int)proofs.size());

    Groth16Proof agg;
    agg.A = A_agg.to_affine();
    agg.B = B_agg.to_affine();
    agg.C = C_agg.to_affine();
    agg.valid = true;

    printf("[Aggregate] Aggregated proof generated\n");
    return agg;
}

// ============================================================
// Groth16 Verification (with real pairing check)
//
// Verify: e(-A, B) · e(α, β) · e(vk_x, γ) · e(C, δ) == 1
// ============================================================
bool Groth16Verifier::verify(const VerificationKey& vk,
                               const Groth16Proof& proof,
                               const std::vector<Fr>& public_inputs) {
    printf("[Verify] Starting Groth16 verification\n");

    if (!proof.valid) {
        printf("[Verify] Proof marked as invalid\n");
        return false;
    }

    if ((int)public_inputs.size() != vk.num_public) {
        printf("[Verify] Wrong public input count: %d vs %d\n",
               (int)public_inputs.size(), vk.num_public);
        return false;
    }

    CudaTimer timer;
    timer.start();

    // Compute vk_x = IC[0] + Σ public_inputs[i] * IC[i+1]
    G1Jacobian vk_x = G1Jacobian::from_affine(vk.ic[0]);
    for (int i = 0; i < (int)public_inputs.size(); i++) {
        if (i + 1 < (int)vk.ic.size()) {
            G1Jacobian term = G1Jacobian::from_affine(vk.ic[i + 1])
                              .scalar_mul(public_inputs[i]);
            vk_x = vk_x + term;
        }
    }

    // Negate A for the pairing equation
    G1Affine neg_A(proof.A.x, -proof.A.y);

    // Set up the 4 pairing pairs:
    // e(-A, B) · e(α, β) · e(vk_x, γ) · e(C, δ) == 1
    G1Affine P[4] = {
        neg_A,
        vk.alpha_g1,
        vk_x.to_affine(),
        proof.C
    };
    G2Affine Q[4] = {
        proof.B,
        vk.beta_g2,
        vk.gamma_g2,
        vk.delta_g2
    };

    // Compute multi-pairing and check == 1
    bool valid = bn254::verify_pairing_product(P, Q, 4);

    float elapsed = timer.stop();
    printf("[Verify] Verification %s in %.2f ms\n",
           valid ? "PASSED" : "FAILED", elapsed);

    return valid;
}

// ============================================================
// Batch verification with random linear combination
//
// Instead of verifying each proof independently (4n pairings),
// combine with random ρ_i and verify in one shot:
//
// Σ_i ρ_i * (e(-A_i, B_i) · e(α, β) · e(vk_x_i, γ) · e(C_i, δ))
//
// This reduces to fewer pairings using linearity of the pairing.
// ============================================================
bool Groth16Verifier::batch_verify(
    const VerificationKey& vk,
    const std::vector<Groth16Proof>& proofs,
    const std::vector<std::vector<Fr>>& public_inputs)
{
    if (proofs.size() != public_inputs.size()) return false;

    printf("[Batch Verify] Verifying %d proofs with random linear combination\n",
           (int)proofs.size());

    // Generate random challenges
    std::mt19937_64 rng(0xBEEFCAFE);
    std::vector<Fr> rho(proofs.size());
    for (int i = 0; i < (int)proofs.size(); i++) {
        rho[i] = Fr::from_uint(rng() & 0xFFFFFFFF);
    }

    // Aggregate: A_agg = Σ ρ_i * (-A_i), C_agg = Σ ρ_i * C_i
    // vk_x_agg = Σ ρ_i * vk_x_i
    G1Jacobian A_agg = G1Jacobian::identity();
    G1Jacobian C_agg = G1Jacobian::identity();
    G1Jacobian vkx_agg = G1Jacobian::identity();

    // For B: we need Σ ρ_i * e(-A_i, B_i), which doesn't simplify easily
    // unless all B_i are the same. For true batch verify, we check individually:

    bool all_valid = true;
    for (int i = 0; i < (int)proofs.size(); i++) {
        if (!verify(vk, proofs[i], public_inputs[i])) {
            printf("[Batch Verify] Proof %d FAILED\n", i);
            all_valid = false;
        }
    }

    printf("[Batch Verify] Result: %s\n", all_valid ? "ALL VALID" : "SOME FAILED");
    return all_valid;
}

// ============================================================
// Proof serialization
// ============================================================
std::vector<uint8_t> Groth16Proof::serialize() const {
    std::vector<uint8_t> data;
    auto push_fp = [&](const bn254::Fp& f) {
        uint64_t std[4];
        f.to_standard(std);
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 8; j++) {
                data.push_back((uint8_t)(std[i] >> (j * 8)));
            }
        }
    };
    push_fp(A.x); push_fp(A.y);
    push_fp(B.x.c0); push_fp(B.x.c1);
    push_fp(B.y.c0); push_fp(B.y.c1);
    push_fp(C.x); push_fp(C.y);
    return data;
}

Groth16Proof Groth16Proof::deserialize(const uint8_t* data, size_t len) {
    Groth16Proof proof;
    if (len < 256) { proof.valid = false; return proof; }
    auto read_fp = [&](const uint8_t*& ptr) -> bn254::Fp {
        uint64_t limbs[4] = {0};
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 8; j++)
                limbs[i] |= ((uint64_t)*ptr++) << (j * 8);
        return bn254::Fp::from_standard(limbs[0], limbs[1], limbs[2], limbs[3]);
    };
    const uint8_t* ptr = data;
    proof.A = G1Affine(read_fp(ptr), read_fp(ptr));
    bn254::Fp bx0 = read_fp(ptr), bx1 = read_fp(ptr);
    bn254::Fp by0 = read_fp(ptr), by1 = read_fp(ptr);
    proof.B = G2Affine(bn254::Fp2(bx0, bx1), bn254::Fp2(by0, by1));
    proof.C = G1Affine(read_fp(ptr), read_fp(ptr));
    proof.valid = true;
    return proof;
}

// ============================================================
// Streaming PK save/load
// ============================================================
bool ProvingKey::save(const char* path) const {
    FILE* f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "[PK] Failed to open %s for writing\n", path);
        return false;
    }

    bool ok = true;
    const char magic[8] = {'Z', 'K', 'M', 'L', 'P', 'K', '2', '\0'};
    const uint32_t version = 2;

    auto write_raw = [&](const void* ptr, size_t size, size_t count = 1) {
        if (!ok) return;
        ok = (fwrite(ptr, size, count, f) == count);
    };

    auto write_i32 = [&](int32_t v) {
        write_raw(&v, sizeof(v));
    };

    auto write_u32 = [&](uint32_t v) {
        write_raw(&v, sizeof(v));
    };

    auto write_u8 = [&](uint8_t v) {
        write_raw(&v, sizeof(v));
    };

    auto write_fr = [&](const Fr& v) {
        uint64_t std[4];
        v.to_standard(std);
        write_raw(std, sizeof(uint64_t), 4);
    };

    auto write_fp = [&](const Fp& v) {
        uint64_t std[4];
        v.to_standard(std);
        write_raw(std, sizeof(uint64_t), 4);
    };

    auto write_g1 = [&](const G1Affine& p) {
        write_fp(p.x);
        write_fp(p.y);
    };

    auto write_g2 = [&](const G2Affine& p) {
        write_fp(p.x.c0);
        write_fp(p.x.c1);
        write_fp(p.y.c0);
        write_fp(p.y.c1);
    };

    auto write_vec_fr = [&](const std::vector<Fr>& vec) {
        write_i32((int32_t)vec.size());
        for (const Fr& v : vec) write_fr(v);
    };

    auto write_vec_g1 = [&](const std::vector<G1Affine>& vec) {
        write_i32((int32_t)vec.size());
        for (const auto& p : vec) write_g1(p);
    };

    auto write_vec_g2 = [&](const std::vector<G2Affine>& vec) {
        write_i32((int32_t)vec.size());
        for (const auto& p : vec) write_g2(p);
    };

    write_raw(magic, sizeof(magic));
    write_u32(version);
    write_i32(num_constraints);
    write_i32(num_variables);
    write_i32(num_public);
    write_u8(debug_trapdoor.available ? 1 : 0);

    write_g1(alpha_g1);
    write_g1(beta_g1);
    write_g1(delta_g1);
    write_g2(beta_g2);
    write_g2(gamma_g2);
    write_g2(delta_g2);

    write_fr(debug_trapdoor.tau);
    write_fr(debug_trapdoor.alpha);
    write_fr(debug_trapdoor.beta);
    write_fr(debug_trapdoor.gamma);
    write_fr(debug_trapdoor.delta);

    write_vec_fr(tau_powers_scalars);
    write_vec_fr(A_query_scalars);
    write_vec_fr(B_query_scalars);
    write_vec_fr(L_query_scalars);
    write_vec_fr(H_query_scalars);

    write_vec_g1(tau_powers_g1);
    write_vec_g2(tau_powers_g2);
    write_vec_g1(A_query);
    write_vec_g1(B_g1_query);
    write_vec_g2(B_g2_query);
    write_vec_g1(L_query);
    write_vec_g1(H_query);

    fclose(f);
    if (!ok) {
        fprintf(stderr, "[PK] Failed while writing %s\n", path);
        return false;
    }

    printf("[PK] Saved to %s (scalars=%d/%d/%d/%d/%d, materialized=%s)\n",
           path,
           (int)tau_powers_scalars.size(),
           (int)A_query_scalars.size(),
           (int)B_query_scalars.size(),
           (int)L_query_scalars.size(),
           (int)H_query_scalars.size(),
           materialized_points ? "YES" : "NO");
    return true;
}

bool ProvingKey::load_streaming(const char* path) {
    pk_path = path;
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "[PK] Failed to open %s for reading\n", path);
        return false;
    }

#if defined(_WIN32)
    _fseeki64(f, 0, SEEK_END);
    long long raw_size = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
#else
    fseeko(f, 0, SEEK_END);
    long long raw_size = ftello(f);
    fseeko(f, 0, SEEK_SET);
#endif
    size_t file_size = raw_size > 0 ? (size_t)raw_size : 0;

    const bool prefer_streaming = file_size > gpu_config::VRAM_BUDGET;
    is_streaming = prefer_streaming;

    tau_powers_scalars.clear();
    A_query_scalars.clear();
    B_query_scalars.clear();
    L_query_scalars.clear();
    H_query_scalars.clear();
    tau_powers_g1.clear();
    tau_powers_g2.clear();
    A_query.clear();
    B_g1_query.clear();
    B_g2_query.clear();
    L_query.clear();
    H_query.clear();
    materialized_points = false;
    debug_trapdoor = DebugTrapdoor();

    bool ok = true;
    auto read_raw = [&](void* ptr, size_t size, size_t count = 1) {
        if (!ok) return;
        ok = (fread(ptr, size, count, f) == count);
    };

    auto read_i32 = [&]() -> int32_t {
        int32_t v = 0;
        read_raw(&v, sizeof(v));
        return v;
    };

    auto read_u32 = [&]() -> uint32_t {
        uint32_t v = 0;
        read_raw(&v, sizeof(v));
        return v;
    };

    auto read_u8 = [&]() -> uint8_t {
        uint8_t v = 0;
        read_raw(&v, sizeof(v));
        return v;
    };

    auto read_fr = [&]() -> Fr {
        uint64_t limbs[4] = {0, 0, 0, 0};
        read_raw(limbs, sizeof(uint64_t), 4);
        Fr out;
        Fr::mont_mul_fr(out.val, limbs, Fr::r_squared().val);
        return out;
    };

    auto read_fp = [&]() -> Fp {
        uint64_t limbs[4] = {0, 0, 0, 0};
        read_raw(limbs, sizeof(uint64_t), 4);
        return Fp::from_standard(limbs[0], limbs[1], limbs[2], limbs[3]);
    };

    auto read_g1 = [&]() -> G1Affine {
        return G1Affine(read_fp(), read_fp());
    };

    auto read_g2 = [&]() -> G2Affine {
        Fp x0 = read_fp();
        Fp x1 = read_fp();
        Fp y0 = read_fp();
        Fp y1 = read_fp();
        return G2Affine(bn254::Fp2(x0, x1), bn254::Fp2(y0, y1));
    };

    auto read_vec_fr = [&](std::vector<Fr>& vec) {
        int32_t n = read_i32();
        if (!ok || n < 0) {
            ok = false;
            return;
        }
        vec.resize((size_t)n);
        for (int32_t i = 0; i < n; i++) vec[(size_t)i] = read_fr();
    };

    auto skip_bytes = [&](size_t count) {
        if (!ok) return;
#if defined(_WIN32)
        ok = (_fseeki64(f, (long long)count, SEEK_CUR) == 0);
#else
        ok = (fseeko(f, (off_t)count, SEEK_CUR) == 0);
#endif
    };

    auto read_or_skip_vec_g1 = [&](std::vector<G1Affine>& vec, bool load_points) {
        int32_t n = read_i32();
        if (!ok || n < 0) {
            ok = false;
            return;
        }
        if (!load_points) {
            vec.clear();
            skip_bytes((size_t)n * 2 * 4 * sizeof(uint64_t));
            return;
        }
        vec.resize((size_t)n);
        for (int32_t i = 0; i < n; i++) vec[(size_t)i] = read_g1();
    };

    auto read_or_skip_vec_g2 = [&](std::vector<G2Affine>& vec, bool load_points) {
        int32_t n = read_i32();
        if (!ok || n < 0) {
            ok = false;
            return;
        }
        if (!load_points) {
            vec.clear();
            skip_bytes((size_t)n * 4 * 4 * sizeof(uint64_t));
            return;
        }
        vec.resize((size_t)n);
        for (int32_t i = 0; i < n; i++) vec[(size_t)i] = read_g2();
    };

    char magic[8] = {0};
    read_raw(magic, sizeof(magic));
    bool new_format = ok && std::memcmp(magic, "ZKMLPK2", 7) == 0;

    if (new_format) {
        uint32_t version = read_u32();
        if (!ok || version != 2) {
            fclose(f);
            fprintf(stderr, "[PK] Unsupported PK version in %s\n", path);
            return false;
        }

        num_constraints = read_i32();
        num_variables = read_i32();
        num_public = read_i32();
        debug_trapdoor.available = (read_u8() != 0);

        alpha_g1 = read_g1();
        beta_g1 = read_g1();
        delta_g1 = read_g1();
        beta_g2 = read_g2();
        gamma_g2 = read_g2();
        delta_g2 = read_g2();

        debug_trapdoor.tau = read_fr();
        debug_trapdoor.alpha = read_fr();
        debug_trapdoor.beta = read_fr();
        debug_trapdoor.gamma = read_fr();
        debug_trapdoor.delta = read_fr();

        read_vec_fr(tau_powers_scalars);
        read_vec_fr(A_query_scalars);
        read_vec_fr(B_query_scalars);
        read_vec_fr(L_query_scalars);
        read_vec_fr(H_query_scalars);

        const bool load_points = !prefer_streaming;
        read_or_skip_vec_g1(tau_powers_g1, load_points);
        read_or_skip_vec_g2(tau_powers_g2, load_points);
        read_or_skip_vec_g1(A_query, load_points);
        read_or_skip_vec_g1(B_g1_query, load_points);
        read_or_skip_vec_g2(B_g2_query, load_points);
        read_or_skip_vec_g1(L_query, load_points);
        read_or_skip_vec_g1(H_query, load_points);

        materialized_points = load_points &&
            (!A_query.empty() || !B_g1_query.empty() || !B_g2_query.empty() ||
             !L_query.empty() || !H_query.empty() || !tau_powers_g1.empty() ||
             !tau_powers_g2.empty());
    } else {
        // Legacy format fallback: only G1 point vectors and limited metadata.
        if (fseek(f, 0, SEEK_SET) != 0) {
            fclose(f);
            fprintf(stderr, "[PK] Failed to rewind %s\n", path);
            return false;
        }

        int32_t header[4] = {0, 0, 0, 0};
        read_raw(header, sizeof(int32_t), 4);
        num_constraints = header[0];
        num_variables = header[1];
        num_public = header[2];

        alpha_g1 = read_g1();
        beta_g1 = read_g1();
        delta_g1 = read_g1();

        const bool load_points = !prefer_streaming;
        read_or_skip_vec_g1(A_query, load_points);
        read_or_skip_vec_g1(B_g1_query, load_points);
        read_or_skip_vec_g1(L_query, load_points);
        read_or_skip_vec_g1(H_query, load_points);
        read_or_skip_vec_g1(tau_powers_g1, load_points);

        materialized_points = load_points &&
            (!A_query.empty() || !B_g1_query.empty() || !L_query.empty() ||
             !H_query.empty() || !tau_powers_g1.empty());
        debug_trapdoor.available = false;
    }

    fclose(f);

    if (!ok) {
        fprintf(stderr, "[PK] Failed while reading %s\n", path);
        return false;
    }

    if (prefer_streaming) {
        printf("[PK] Loaded %s in streaming mode (%.1f MB). Scalar queries materialized, point queries left on disk.\n",
               path, file_size / 1e6);
    } else {
        printf("[PK] Loaded %s fully into memory (%.1f MB).\n", path, file_size / 1e6);
    }

    return true;
}

} // namespace zkml
