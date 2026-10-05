# Experiment Plan — protocol-v2 publication validation

The first T4 validation run is complete. This plan now describes the **next replication gate** for the single selected-proof protocol.

## Gate A — clean NVIDIA build and regression

1. clean Release CMake configure/build;
2. `pytest -q`;
3. `ctest --output-on-failure`;
4. record GPU, driver, CUDA toolkit and build flags.

## Gate B — exact arithmetic correctness

For every candidate path:

1. generate/load its model-specific Groth16 key;
2. prove a benchmark sample;
3. native-verify the proof;
4. require exact CUDA/Python integer-logit equality;
5. fail closed on any mismatch.

## Gate C — measured path costs

Run at least five proof repeats per path and record median/p95/min/max prover time, verifier time, inference time, proof size, R1CS variables and constraints.

## Gate D — protocol-v2 held-out evaluation

- Calibrate thresholds only on the calibration split.
- Use measured standalone prover time as each path cost.
- Use `selected_proof` cost semantics.
- Evaluate once on the held-out test split.
- Report adaptive/full accuracy, accuracy drop, expected selected-proof cost, static-full proof cost, savings and route counts.

## Gate E — single-proof route evidence

For representative held-out routes:

- generate only the selected model-specific proof;
- pin its verification key in the protocol-v2 manifest;
- native-verify the proof;
- check model tag and proof-bound public scores;
- require the selected path's margin threshold to pass (unless full fallback).

## Gate F — publication-strength replication

Implemented; run `scripts/run_gate_f.sh` (see `docs/GATE_F_RUNBOOK.md`).

- repeat on at least one additional NVIDIA GPU;
- add at least one larger benchmark/model family;
- compare matched statement semantics against relevant contemporary zkML systems;
- run systematic literature/patent novelty audit;
- seek external cryptographic review.

## Success criterion

A replication supports the PCANI v2 hypothesis only if the predeclared accuracy budget is met **and** expected measured selected-proof cost is lower than static full-model proving.
