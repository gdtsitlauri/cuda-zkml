# CUDA-zkML / PCANI Research Release Manifest

**Release:** 2026-09 single selected-proof research release  
**Version:** `2026.3-single-proof`

## Included research components

- CUDA BN254/Groth16 implementation and hardened proving-key handling.
- Exact integer fixed-model R1CS statements.
- Model-specific verification-key digest and public model-tag binding.
- Protocol v1 proof-chain verifier retained for reproducibility.
- Protocol v2 single selected-proof verifier.
- Deterministic `sklearn Digits` benchmark.
- Completed NVIDIA T4 validation artifacts under `results/gpu_validation_2026/`.
- Protocol-v2 re-analysis under `results/gpu_validation_2026/single_proof_v2/`.
- Research paper source/PDF.

## Validation status

The successful T4 run produced real repeated prover timings, exact CUDA/Python output checks, native Groth16 verification for representative route bundles, and held-out adaptive accuracy measurements.

The original protocol-v1 hypothesis failed on cost because cumulative proof-chain proving averaged 2899.95 ms versus 1884.02 ms for the static full path.

The protocol-v2 single-proof redesign yields 1602.51 ms expected selected-proof cost from the same measured per-path T4 medians and held-out route counts, a 14.94% reduction at a 0.83 pp accuracy drop.

## Claims intentionally not made

- no production cryptographic certification;
- no private-input adaptive routing;
- no model hiding;
- no universal zkML speedup claim;
- no claim that v2 proves the selected route is globally cheapest;
- no established PhD novelty claim before systematic literature/patent review.

## Reproduction

- `python/pcani_single_proof_finalize.py results/gpu_validation_2026` reproduces the protocol-v2 re-analysis from the frozen GPU artifacts.
- `PCANI_Colab_Final_Validation.ipynb` remains the original protocol-v1 GPU experiment notebook and is kept for reproducibility/history.
