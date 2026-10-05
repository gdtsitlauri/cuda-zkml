# PCANI protocol v2 - single selected-proof routing

## Why v2 exists

The real T4 run showed a structural failure in protocol v1: requiring a valid proof for every rejected cheap path makes the adaptive system pay a large fixed Groth16 cost multiple times. The policy was accurate, but cumulative proving overhead erased the benefit.

Protocol v2 separates **route validity** from **route minimality**.

## Protocol

For a selected path `Pk`, the prover sends exactly one model-specific proof. The verifier checks:

1. the certificate policy digest matches the pinned policy;
2. the chosen path is permitted by the manifest;
3. the verification-key SHA-256 matches the manifest;
4. native Groth16 verification succeeds;
5. the public model tag matches the model-specific circuit;
6. the proof-bound public input/output layout is valid;
7. for every non-final path, the proof-bound top-1/top-2 margin clears that path's calibrated threshold;
8. the final path remains an unconditional fallback.

The verifier does **not** demand evidence that every cheaper path would have rejected the same input. Therefore v2 proves that the selected route is valid under the calibrated acceptance rule, not that it is globally minimal. Selecting the cheapest valid path is an optimization objective of the prover/policy rather than a cryptographic soundness condition.

## Why this preserves the core guarantee

A malicious prover cannot forge a smaller-path result merely by claiming a route: the selected model proof must verify and its public scores must satisfy the acceptance rule. Choosing a more expensive valid path does not violate inference correctness; it only hurts the prover's own cost.

## Measured effect on the 2026 T4 run

The v1 chain cost was 2899.95 ms expected versus 1884.02 ms static. Under v2, using the same measured standalone prover medians and the same held-out routing decisions, expected selected-proof cost is 1602.51 ms, a 14.94% reduction.

## Current privacy boundary

The input and output scores are public in protocol v2. The exact model is bound by the fixed circuit and model-specific verification key, but model hiding is not claimed. Private input commitments and private routing predicates are future work.
