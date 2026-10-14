# CUDA-zkML / PCANI — Roadmap for future work (PhD)

Status: 2026-10-03. The project is frozen here and continues in the doctorate.
This file states **what remains, in which order, where in the code and how it is checked**.

## Update 2026-10-04: Phase 1 and Phase 2 are implemented; only runs remain

The code for every goal of Phase 1 and Phase 2 is written. The part that needs no GPU
(circuits, witness, Poseidon, ceremony, Python) was checked locally. The full GPU chain
(setup, proof, verify) runs with **one notebook**: `notebooks/PCANI_Colab_Phase12.ipynb`,
which calls `scripts/run_phase12.sh`.

| Goal | Implementation | Local check (no GPU) | Remaining |
|---|---|---|---|
| 1.1 context / anti-replay | `--statement-v2 --context`, `ctx·ctx=ctx_sq` (`src/prover/gadgets.cuh`), check of `expected_context` / `seen_contexts` (`python/zkml/pcani_protocol_v2.py`), `contracts/PCANIContextVerifier.sol` | ✅ host + pytest; fixed **input aliasing** in `Verifier.sol` (check < q instead of < r) | run E1 |
| 1.2 underconstrained | export to `.r1cs`/`.wtns` (iden3, `src/prover/r1cs_export.cuh`), perturbation check in C++, independent check `tests/test_underconstrained.py` | ✅ 0 free variables (8,082 and 62,692 constraints) | E6; optionally Picus/SMT |
| 1.3 threat model | `docs/THREAT_MODEL.md` | ✅ | — |
| 2.1 ReLU with range proofs | `RELU_EXACT` (bits + rescale), `python/export_trained_relu_model.py` | ✅ trained digits: float 97.3% → integer 97.1%, C++ = Python bit for bit | E1 |
| 2.2 native conv | `CONV2D` (sparse rows, only non-zero taps) | ✅ | E2 |
| 2.3 private model | `--private-model`, Poseidon BN254 (**same as circomlib**: reference vector and `poseidon([1,2])`) | ✅ | E4 |
| 2.4 private input | `--private-input --input-blinding`, routing inside the circuit (top-1/top-2, flag), `--private-outputs` | ✅ | E4 |
| 2.5 MPC ceremony | `zkml-ceremony contribute/verify` (`src/prover/ceremony.cuh`), PoK + pairings | ✅ exact algebra + rejection of tampering (9/9) | E5; real participants (Phase 3); phase 1 stays single-party |
| 2.6 batching | `--batch-inputs` | ✅ | E3 (timings) |

Running now: `notebooks/PCANI_Colab_Phase12.ipynb` (any NVIDIA GPU) → `Phase12_<gpu>_<date>.zip`.
Locally, without CUDA: `tests/host/run_host_tests.sh` and `pytest`.
The v1 path (the published T4 results) **did not change** and reproduces as before.

Basic principle: security must rest only on the keys and the cryptographic assumptions,
not on the secrecy of the code (Kerckhoffs). The goal is not "zero weaknesses" but an
explicit threat model, a defence for everything inside it, and a clear statement of
what stays outside.

---

## Phase 0 — Runs only (all ready)

Nothing needs code. Details: `docs/GATE_F_RUNBOOK.md`.

| Step | How |
|---|---|
| Repeat protocol v2 on a 2nd GPU (not T4) | `notebooks/PCANI_Colab_GateF.ipynb` (L4/A100) or `bash scripts/run_gate_f.sh` |
| MNIST-MLP and MNIST-CNN | same script (`WORKLOADS="mnist_mlp mnist_conv"`) |
| Fair comparison with EZKL | same script (`EZKL=1`) |

Output: `GateF_<gpu>_<date>.zip` → numbers for the first paper.

Possible issues: memory on a 4 GB card for `mnist_conv p12`; if the path timings do not
come out increasing, raise `REPEATS`.

---

## Phase 1 — Before the first paper (small changes)

### 1.1 Binding the proof to a context (anti-replay)
- **Problem:** Groth16 proofs are malleable, and a valid proof can be reused for another
  request.
- **Solution:** an extra public input `context = H(session_id ‖ nonce ‖ timestamp ‖ policy_digest)`
  bound by a trivial constraint in the circuit, like `model_tag`.
- **Files:** `src/cli/prove.cu` (`build_fixed_model_circuit`, statement meta),
  `src/prover/witness.cuh`, `python/zkml/pcani_protocol_v2.py` (context check in the
  certificate), `contracts/Verifier.sol` (+1 public input).
- **Check:** a test where a proof with a wrong context is rejected.

### 1.2 Check for an underconstrained circuit
- **Problem:** the most common class of real bugs in zkML. The circuit may accept a
  witness that does not correspond to the correct inference.
- **Solution:** (a) export the R1CS to a file (e.g. circom's `.r1cs` format),
  (b) negative tests: a tampered witness must fail `R1CS::verify`, (c) a check that every
  output variable is uniquely determined (tools such as Picus / Circomspect, or our own
  SMT check on small circuits).
- **Files:** `src/prover/circuit.cuh`, new `tests/test_underconstrained.py`.

### 1.3 Explicit threat model in the paper
- The model is **public** (constants inside the circuit → recoverable from pk/vk).
- The input is **public**; the routing is **public**.
- The per-model trusted setup is a **trust assumption**.

---

## Phase 2 — Doctoral extensions

### 2.1 Real ReLU with range proofs (the main technical gap)
- **Today:** activation `h = z² + 64z + 64` and random, untrained hidden weights
  (`python/pcani_prepare.py`).
- **Solution:** decompose z into bits inside the circuit (k constraints for a k-bit range)
  → sign bit s → `relu(z) = (1 - s)·z`. Then trained multi-layer models with quantization
  (int8/int16) and rescaling through range-checked division.
- **Files:** new `LayerType::RELU_EXACT` in `src/nn/layers.cuh`, `inference.cu`,
  `circuit.cuh` (`add_relu_exact`), `witness.cuh`; `python/zkml/model.py` and a new
  exporter of trained models (PyTorch).
- **Check:** exact equality of CUDA/Python logits (as today) on a trained MNIST-CNN, a
  negative test with a wrong bit.

### 2.2 Native convolution layer
- **Today:** the CNN is unrolled into a dense matrix with many zeros.
- **Solution:** `LayerType::CONV2D` with sparse linear constraints (only the non-zero
  weights). In parallel, skip zero terms in `add_fixed_linear_layer_into` (careful: this
  changes the R1CS of Digits, so it is a separate experiment and does not replace the T4
  result).

### 2.3 Private model (protection against weight extraction / reverse engineering)
- **Today:** anyone with pk/vk can read all the weights.
- **Solution:** weights as a private witness + a public commitment
  `C_model = Poseidon(weights)` computed inside the circuit. Needs a Poseidon (BN254)
  circuit in R1CS and a matching witness generator. One universal pk/vk per
  **architecture**, not per model.
- **Cost:** the weight×input multiplications become real constraints (today they are
  "free" linear ones) → a much larger circuit. A research question in itself (cost of
  privacy vs PCANI savings).

### 2.4 Private input
- Same mechanism: `C_input = Poseidon(x ‖ r)` public, x private.
- The routing (path choice) must then move **inside** the circuit (comparison of the
  margin with the threshold through a range proof), otherwise it leaks information.

### 2.5 Trusted setup
- **Today:** one setup per model; whoever runs it can forge.
- **Realistic solution:** an MPC phase-2 ceremony tool for Groth16 (each participant
  multiplies δ by a random s and updates the L/H queries, with a consistency proof).
  Files: `src/prover/groth16.cu` (ProvingKey), new CLI `zkml-ceremony`.
- **Large alternative:** a universal (PLONK/KZG) or transparent (STARK/Halo2) setup →
  essentially a new prover; only if it becomes the central topic of the PhD.

### 2.6 Scale
- Batching many samples in one proof; larger models (a small ResNet, a small
  transformer — `SELF_ATTENTION` already exists in the circuit).

---

## Phase 3 — Not solvable with code

| Topic | What it needs |
|---|---|
| Systematic literature/patent novelty audit | reading — keywords in `docs/NOVELTY_GATE_2026.md` |
| External cryptographic review | a cryptographer (collaborator) |
| MPC ceremony | real participants |
| Side channels of the GPU prover (once the witness is private) | timing/memory measurements; to be stated in the threat model |

---

## Tools needed for Phase 1–2

- Linux or **WSL2 + Ubuntu** with the CUDA toolkit (a GTX 1650 / sm_75 is enough for
  development); Colab for the final runs.
- Python: `numpy scikit-learn pytest onnx ezkl` (+ `torch` for trained models).

## Suggested order

1. Phase 0 (runs) → first paper.
2. 1.1 + 1.2 + 1.3.
3. 2.1 (ReLU) → 2.2 → second paper (trained models).
4. 2.3 + 2.4 + 2.5 → third paper (private PCANI).
