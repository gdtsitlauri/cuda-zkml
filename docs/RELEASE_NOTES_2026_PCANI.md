# CUDA-zkML 2026 PCANI Single-Proof Research Release

## Release summary

This release freezes the first complete GPU-validated PCANI research cycle.

### What changed since the proof-chain RC

- Fixed the Groth16 `C`-element convention bug for the prover's unblinded `B1` representation.
- Fixed signed integer conversion from BN254 `Fp` into `Fr`; this bug was exposed by the exact CUDA/Python output gate.
- Preserved the protocol-v1 proof-chain result as a negative baseline.
- Added protocol-v2 single selected-proof route certificates.
- Added selected-proof cost semantics to policy calibration/evaluation.
- Added `pcani_single_proof_finalize.py` to re-analyze a completed GPU run from stored artifacts without inventing timings.
- Added GPU-result documentation and the uploaded validation artifacts.

## Measured T4 result

Median prover time (5 repeats):

- p16: 1475.75 ms
- p48: 1628.66 ms
- p96: 1781.65 ms
- p160: 1884.02 ms

Held-out adaptive accuracy: 96.11%  
Static p160 accuracy: 96.94%  
Accuracy drop: 0.83 percentage points.

Protocol v1 cumulative chain expected proof cost: 2899.95 ms (negative result).

Protocol v2 selected-proof expected cost, using the same measured per-path medians and held-out routes: 1602.51 ms, a 14.94% proof-cost reduction relative to static p160.

## Evidence boundary

The v2 cost figure is a weighted re-analysis of measured standalone prover medians and observed held-out routing. Representative chosen-path proofs were native-Groth16 verified in the original T4 run. It is not a fresh monolithic v2 timing run and not every held-out sample was individually proved.

## Remaining gates

- multi-GPU / multi-benchmark replication;
- matched contemporary zkML comparisons;
- systematic literature/patent novelty audit;
- external cryptographic review before production claims.
