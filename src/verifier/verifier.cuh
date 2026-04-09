#pragma once

#include "prover/groth16.cuh"

// ============================================================
// Native Groth16 Verifier
//
// Verification equation:
//   e(A, B) = e(α, β) · e(L, γ) · e(C, δ)
//
// where L = IC[0] + Σ public_inputs[i] * IC[i+1]
//
// Cost: 4 pairings (or 1 multi-pairing with optimized Miller loop)
// This is constant regardless of circuit size.
//
// For EVM verification, see contracts/Verifier.sol
// ============================================================

namespace zkml {

// File-based proof I/O
bool save_proof(const Groth16Proof& proof, const char* filename);
Groth16Proof load_proof(const char* filename);

bool save_vk(const VerificationKey& vk, const char* filename);
VerificationKey load_vk(const char* filename);

bool save_public_inputs(const std::vector<bn254::Fr>& inputs, const char* filename);
std::vector<bn254::Fr> load_public_inputs(const char* filename);

} // namespace zkml
