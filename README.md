# CUDA-zkML / PCANI

**Can a service prove which neural network produced its answer — cheaply enough, without replay, and without revealing the model or the input?**

CUDA-zkML is a GPU (CUDA) Groth16 prover for zero-knowledge proofs of quantized neural-network inference on
BN254. The field arithmetic, MSM, NTT and pairing are written for the GPU. A proof is 256 bytes and verifies in
milliseconds, natively or in a Solidity contract.

**PCANI** (proof-cost-aware adaptive inference) adds a cascade of models from cheap to expensive:
- if the cheap model is confident, only its proof is produced;
- otherwise the next model is used;
- the routing decision itself is proof-bound.

The 2026-10 work adds **statement v2**:
- proofs bound to a session context (anti-replay);
- exact ReLU with range proofs, and native convolution;
- a private model and a private input behind Poseidon commitments, with routing computed inside the circuit;
- batching and a Groth16 phase-2 MPC ceremony.

| part | content | evidence |
|---|---|---|
| prover | GPU Groth16 (MSM, NTT, pairing), group-only proving keys, Solidity verifier | ctest, T4 validation run |
| PCANI v1/v2 | proof-chain vs single selected proof, pinned model tags and keys | completed T4 experiment |
| statement v2 | context binding, RELU_EXACT, CONV2D, Poseidon commitments, in-circuit routing, batching | host tests, Python tests, Colab notebook |
| setup | `zkml-ceremony` phase-2 MPC | host tests |

## Main findings

GPU numbers come from the completed NVIDIA T4 run (`results/gpu_validation_2026/`). Statement-v2 numbers come
from host (no-GPU) tests of the exact circuits the prover uses; the GPU end-to-end run is pending (see below).

1. **Proving a fixed chain of cascade models does not pay; one selected proof does.**
   - Protocol v1 (prove every tried model): 96.11% accuracy, but the expected proof cost was 2899.95 ms against
     1884.02 ms for always proving the large model. The hypothesis is **rejected** (−53.9%).
   - Protocol v2 (one proof for the selected model, with its proof-bound margin checked against the calibrated
     threshold): **1602.51 ms, a 14.94% reduction**, at the same 96.11% accuracy (0.83 pp below the large
     model). Measured on the same per-path prover medians and the frozen held-out routing trace.
2. **Proofs can be bound to a session (anti-replay).** A public context, H(session, nonce, time, policy), is tied
   into the R1CS. A proof for another session or a replayed proof is rejected, natively and on chain
   (`PCANIContextVerifier.sol`). The work also found and fixed **public-input aliasing in `Verifier.sol`**: inputs
   were bounded by the base field q instead of the scalar field r, so x and x + r verified alike.
3. **The circuits are not underconstrained.** The circuit is exported in the iden3 `.r1cs` format. An independent
   Python checker perturbs every variable and finds **0 unpinned variables**: 8,082 constraints in the
   public-model statement and 62,692 constraints / 62,779 variables in the private one. Forged witnesses are
   rejected, for example `relu(-9) = -9` and out-of-range inputs.
4. **Trained models with an exact ReLU.** The old polynomial activation with random weights is replaced by a
   bit-decomposed ReLU with range-checked rescaling. A trained digits MLP reaches 97.3% in float and **97.1% as
   the exact integer model the circuit proves**. C++ and Python scores agree bit for bit (batch of 8).
5. **Private model and private input.** Weights and inputs are hidden behind Poseidon commitments. The Poseidon
   implementation equals the circomlib instance (reference test vector, `poseidon([1,2])`). The top-1/top-2
   routing comparison runs inside the circuit, so only the route bit and the class are public.
6. **Trusted setup can be distributed.** `zkml-ceremony` runs Groth16 phase 2: each contribution rescales δ and
   the L/H queries and publishes a proof of knowledge. Every contribution is checked with pairings, and tampered
   keys, transcripts or verification keys are rejected (9/9 checks).

## Negative and limiting results (reported as such)

- Protocol v1 failed its cost objective (above). It is kept as the negative baseline.
- Protocol v2 proves that the selected route is valid, not that it was the cheapest valid route.
- The 14.94% saving is from one GPU (T4) and one dataset (Digits). Replication on another GPU, MNIST and a matched
  EZKL comparison are prepared (`PCANI_Colab_GateF.ipynb`) but not run.
- Statement v2 has not yet been run end to end on a GPU; its proving times are unknown.
- Privacy is expensive: the private statement needs about 8x more constraints than the public one. Whether this
  outweighs the PCANI saving is an open research question.
- Setup phase 1 (powers of tau) is still single-party; the ceremony covers phase 2 only.
- Proofs certify the quantized integer model, not the float model. The architecture is always public.
- Historical EZKL/Orion rows are engineering measurements, not matched comparisons.
- No external cryptographic audit and no systematic novelty audit have been done.

## Folder map

```
CUDA-zkML-PCANI-SingleProof-2026/
  README.md, LICENSE (MIT), FUTURE_WORK_ROADMAP.md (status + what remains)
  SUMMARY_FOR_SUPERVISOR_GR.txt   plain-language summary of the whole study (Greek)
  src/
    field/, curve/, msm/, ntt/    BN254 arithmetic, pairing, Pippenger MSM, NTT (CUDA)
    nn/                           quantized inference incl. RELU_EXACT and CONV2D
    prover/                       Groth16, R1CS, v1 witness, statement v2, gadgets, Poseidon, r1cs export,
                                  MPC ceremony
    cli/                          zkml-prove, zkml-verify, zkml-ceremony
  contracts/                      Verifier.sol, PCANIContextVerifier.sol
  python/zkml/                    PCANI protocols v1/v2, statement v2, Poseidon reference, model tools
  python/                         experiment drivers, trained-model exporter, phase-1/2 end-to-end driver
  tests/                          CUDA tests, Python tests, underconstraint checker
  tests/host/                     no-GPU tests: gadgets, statement v2, MPC ceremony (+ CUDA shim)
  docs/                           threat model, protocol v2, GPU validation results, technical details
  results/gpu_validation_2026/    T4 run: per-path artifacts, proofs, route bundles, summaries
  benchmarks/                     benchmark results and baselines
  PCANI_Colab_*.ipynb             Final validation (done), Gate F, Phase 1-2 end-to-end
```

`docs/TECHNICAL_DETAILS.md` has build options, the CLI and Python API, and implementation details (MSM, NTT,
pairing, Solidity export).

## Reproducing

| result | command | where |
|---|---|---|
| 1 | `PCANI_Colab_Final_Validation.ipynb` | Colab GPU |
| 2–6 (logic) | `tests/host/run_host_tests.sh` and `pytest` | any CPU, no CUDA |
| 2–6 (GPU, end to end) | `PCANI_Colab_Phase12.ipynb` (runs `scripts/run_phase12.sh`) | Colab GPU, ~1–2 h |
| replication | `PCANI_Colab_GateF.ipynb` | Colab GPU (not T4), ~2–3 h |

Build locally: `cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=<sm>` and `cmake --build build`; then `ctest --test-dir build`.

## Status and what remains

Roadmap Phases 1 and 2 are implemented. What remains is only runs and external work (`FUTURE_WORK_ROADMAP.md`):

- **Runs:** `PCANI_Colab_Phase12.ipynb` (GPU end to end for statement v2) and `PCANI_Colab_GateF.ipynb`
  (second GPU, MNIST, matched EZKL).
- **External:** a ceremony with real participants, a cryptographer's review and a systematic novelty audit.

## Protocol discipline

The T4 experiment ran fail-closed:
- exact agreement of CUDA and Python scores;
- native verification of every proof;
- pinned verification-key digests and model tags.

Protocol v1 was rejected on those measured numbers and is kept, not hidden. Protocol v2 reuses the same prover
medians and the frozen routing trace.

The v1 statement path is unchanged by the v2 work, so the published results reproduce. Reference values for
the v2 work come from independent code paths:
- the Python Poseidon (checked against the circomlib vector);
- the Python exact-integer inference;
- a pure-Python R1CS checker.

The draft paper was removed; it will be rewritten after the GPU runs.

## Citation and license

George David Tsitlauri, University of Thessaly, 2026. MIT License (`LICENSE`).
