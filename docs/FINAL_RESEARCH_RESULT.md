# Final 2026 research result

The first completed NVIDIA T4 experiment produced a useful two-stage result.

## Protocol v1 — rejected

The original proof-chain design kept routing fully replayable by proving every rejected cheap path before the chosen path. It met the quality budget but failed the cost objective:

- adaptive held-out accuracy: **96.11%**
- static p160 accuracy: **96.94%**
- accuracy drop: **0.83 percentage points**
- expected proof-chain cost: **2899.95 ms**
- static p160 proof cost: **1884.02 ms**
- proof-cost savings: **-53.92%**

## Protocol v2 — single selected proof

Protocol v2 sends only the selected model-specific proof. The verifier checks the pinned model/key and verifies that the proof-bound score margin satisfies the selected path's calibrated acceptance rule.

Using the same measured T4 per-path prover medians and the frozen held-out routing trace:

- expected selected-proof cost: **1602.51 ms**
- static p160 proof cost: **1884.02 ms**
- proof-cost reduction: **14.94%**
- proof-cost speedup: **1.176x**
- adaptive accuracy: **96.11%**
- accuracy drop: **0.83 pp**
- component-derived inference + proof estimate: **1737.06 ms** vs **1958.48 ms** static
- estimated end-to-end reduction: **11.31%**

## Safe claim

The current repository supports a hardened CUDA/Groth16 zkML research system and a proof-cost-aware adaptive inference protocol whose first cumulative proof-chain design failed on measured cost, while the single-selected-proof redesign produces a positive expected proof-cost/accuracy trade-off on the completed T4 Digits experiment.

## Still required before a strong publication / PhD novelty claim

- replication on another GPU;
- at least one larger benchmark/model family;
- matched contemporary zkML comparisons;
- systematic 2026 literature/patent novelty audit;
- external cryptographic review.
