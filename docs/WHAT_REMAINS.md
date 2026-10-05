# What remains - plain language

The core 2026 PCANI research prototype is now implemented and the first NVIDIA T4 validation run is complete.

## Completed

- hardened Groth16 proving-key persistence;
- pairing-consistent Groth16 proof generation;
- correct signed `Fp -> Fr` integer mapping;
- exact fixed-model R1CS statements;
- model-specific verification-key pinning;
- exact CUDA/Python output checks;
- deterministic Digits benchmark;
- five repeated prover measurements for p16/p48/p96/p160;
- protocol-v1 proof-chain evaluation (negative cost result preserved);
- protocol-v2 single selected-proof verifier and cost semantics;
- held-out v2 re-analysis using the measured T4 path costs;
- representative cryptographic route evidence.

## What remains for publication / strong PhD claim

Items 1–3 are implemented and only need a GPU run (`scripts/run_gate_f.sh` or
`PCANI_Colab_GateF.ipynb`; details in `docs/GATE_F_RUNBOOK.md`).

1. Repeat protocol v2 on at least one additional NVIDIA GPU. — *run only*
2. Add at least one larger benchmark/model family beyond Digits. — *run only* (MNIST MLP + lowered-convolution family)
3. Run matched comparisons against relevant contemporary zkML systems where statement semantics are comparable. — *run only* (`run_ezkl_matched.py`; preliminary CPU rows in `results/ezkl_matched_preliminary_2026-10-03/`)
4. Perform the systematic 2026 literature and patent novelty audit.
5. Obtain external cryptographic review before any production-security claim.
6. Optionally add private-input commitments and hidden routing if privacy, rather than verifiable public inference, becomes part of the contribution.

## Current safe claim

The current evidence supports: a hardened CUDA/Groth16 zkML research system plus a proof-cost-aware adaptive inference protocol whose first proof-chain design failed on measured cost, and whose single-selected-proof redesign yields a 14.94% lower expected prover cost on the held-out Digits routing trace using measured T4 per-path prover medians, at a 0.83 percentage-point accuracy drop.

It does **not** yet support: universal speedup, production security, private adaptive inference, or established novelty against all 2026 literature/patents.
