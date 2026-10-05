# GPU validation results - 2026 T4 run

## What was actually measured

The uploaded `PCANI_final_results.zip` is the evidence source for this release. The successful NVIDIA T4 run used the deterministic exact-integer `sklearn Digits` benchmark, four candidate paths (`p16`, `p48`, `p96`, `p160`), five repeated Groth16 prover measurements per path, and a held-out test split of 360 samples.

All four paths passed the exact CUDA/Python logit check. Representative route bundles passed native Groth16 verification, model-specific verification-key pinning, fixed-model tag checks, same-public-input linkage, and public margin-routing checks.

| Path | Test accuracy | Median prover | Median inference | Constraints | Proof size |
|---|---:|---:|---:|---:|---:|
| p16 | 79.44% | 1475.75 ms | 72.09 ms | 75 | 256 B |
| p48 | 93.61% | 1628.66 ms | 73.36 ms | 203 | 256 B |
| p96 | 96.39% | 1781.65 ms | 74.65 ms | 395 | 256 B |
| p160 | 96.94% | 1884.02 ms | 74.46 ms | 651 | 256 B |

The calibrated thresholds were `350937070.4`, `254501196.0`, and `237925533.0` for p16/p48/p96. On held-out data the route counts were p16=146, p48=143, p96=51, p160=20.

## Protocol v1 result: negative

Protocol v1 required a prefix proof chain. Its held-out adaptive accuracy was 96.11% versus 96.94% for p160, an accuracy drop of 0.83 percentage points, which satisfied the predeclared 1 pp held-out tolerance.

However, its cumulative expected proof cost was **2899.95 ms**, compared with **1884.02 ms** for a single static p160 proof. That is **-53.92% proof-cost savings** (i.e. a 53.92% increase). The v1 performance hypothesis is therefore rejected on this run.

This negative result is preserved; it is not hidden or overwritten.

## Protocol v2 result: single selected proof

Protocol v2 does not require proofs for cheaper rejected paths. The prover evaluates the deterministic policy, emits only the selected model-specific proof, and the verifier accepts the route only if that proof verifies and its proof-bound margin satisfies the selected path's calibrated threshold (unless the selected path is the unconditional full fallback).

Reusing the measured standalone median prover times from the same T4 run, the held-out expected selected-proof cost is:

- **1602.51 ms** adaptive selected-proof cost
- **1884.02 ms** static p160 proof cost
- **14.94% proof-cost reduction**
- **1.176x proof-cost speedup**
- **96.11% adaptive accuracy**
- **0.83 pp accuracy drop** versus p160

Using the measured median inference times for every sequentially evaluated candidate, the component-derived end-to-end estimate is:

- adaptive inference + proof: **1737.06 ms**
- static p160 inference + proof: **1958.48 ms**
- estimated reduction: **11.31%**
- estimated speedup: **1.127x**

The 14.94% proof-cost figure is a direct weighted combination of measured path prover medians and observed held-out route counts. The 11.31% end-to-end figure is derived from measured component medians. No new timing was fabricated.

## Evidence boundary

The v2 re-analysis does **not** mean every one of the 360 held-out samples was separately proved. The run measured each path repeatedly and cryptographically verified representative route bundles. Protocol v2 reuses the already verified chosen-path proofs from those representative bundles.

Before publication-quality claims, repeat the v2 experiment on at least one additional NVIDIA GPU and additional benchmarks, and run an external cryptographic review.
