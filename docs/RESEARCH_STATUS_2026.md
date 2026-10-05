# CUDA-zkML / PCANI Research Status - September 2026

## Status

**GPU validation completed. Protocol v1 rejected on cost. Protocol v2 single-proof redesign implemented and re-analyzed from the measured run.**

## Hardened foundation

- Group-only persisted Groth16 proving keys; toxic setup scalars are not serialized.
- Legacy secret-bearing key formats fail closed.
- Unsupported linear aggregation is disabled.
- Groth16 `C` is consistent with the prover's unblinded-`B1` convention.
- Signed integer semantics are preserved across BN254 base/scalar field conversion.
- Exact fixed-model statements bind quantized parameters to the circuit.
- Verification-key digests and public model tags pin the permitted model.

## T4 evidence

- Exact CUDA/Python outputs: PASS for all four candidate paths.
- Native Groth16 verification: PASS for representative route bundles.
- Five repeated prover measurements per path.
- Held-out adaptive accuracy: 96.11%.
- Static p160 accuracy: 96.94%.
- Accuracy drop: 0.83 pp.

Protocol v1 cumulative chain: 2899.95 ms expected proof cost -> hypothesis not supported.

Protocol v2 selected proof: 1602.51 ms expected proof cost from measured path medians -> 14.94% lower than static p160's 1884.02 ms. Component-derived inference+proof reduction: 11.31%.

## Claim boundary

The v2 cost result is a re-analysis of real measured path timings and real held-out routing, not a fabricated or simulated timing. It is not a fresh monolithic v2 GPU benchmark and does not prove every held-out sample individually. Broader replication and novelty review remain mandatory before paper-level novelty claims.
