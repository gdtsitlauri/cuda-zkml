# Threat model: CUDA-zkML / PCANI (roadmap 1.3)

## Principle

The adversary knows all code, every circuit, and every public artifact (Kerckhoffs).
Security rests only on:

- the Groth16 knowledge-soundness and zero-knowledge assumptions on BN254;
- the collision resistance and hiding of Poseidon;
- SHA-256;
- the trust assumption of the setup.

## Parties

| Party | Trust |
|---|---|
| Prover (runs the model on GPU, produces proofs) | untrusted |
| Verifier / smart contract | trusted to run its own checks |
| Setup participants | at least one honest, see "Setup" |
| Model owner (private-model mode) | owns the weights and publishes `C_model` |
| Data owner (private-input mode) | owns `x`, publishes `C_input` |

## Statements and what they protect

| Mode | Public | Private | Guarantees | Does **not** hide |
|---|---|---|---|---|
| v1 fixed model (published T4 results) | scores, model tag, input | none | correct inference of the exact quantized model on that input | model (it can be read from pk/vk), input |
| v2 + `--context` | same + context | none | the above, plus binding to one session (anti-replay) | model, input |
| v2 + `--private-model` | `C_model` | weights | correct inference of *some* model whose commitment is `C_model`; one key serves the whole architecture | architecture, layer sizes |
| v2 + `--private-input` | `C_input = Poseidon(x ‖ r)` | `x`, blinding `r` | the scores belong to the committed input | output scores (unless `--private-outputs`) |
| v2 + `--route-threshold` [+ `--private-outputs`] | route bit [top1 − top2 ≥ τ] and top-1 class | scores | routing is computed **inside** the circuit; nothing leaks besides the decision | the decision itself |

## In scope: attacks and defences

| # | Attack | Defence | Evidence |
|---|---|---|---|
| Z1 | Proof for a different model than the one claimed | Weights are circuit coefficients (v1/v2), or `C_model` with Poseidon (private) | host tests; E1/E4 in `run_phase12_e2e.py` |
| Z2 | Wrong or forged inference result | R1CS correctness; exact ReLU via bits; range-checked rescaling | `test_gadgets` (forged `relu(-9) = -9` rejected); E1 altered score rejected |
| Z3 | **Underconstrained circuit** (witness ≠ real inference) | Every variable is pinned by the constraints | `test_gadgets` / `test_statement_v2` (single-variable perturbation over 62,779 variables); independent check in `tests/test_underconstrained.py` on the exported `.r1cs` |
| Z4 | Replay of a valid proof in another session | Public context bound in the R1CS (`ctx·ctx = ctx_sq`); verifier compares `expected_context` and `seen_contexts`; on-chain `PCANIContextVerifier` (freshness, single use) | `test_statement_v2_protocol.py`; E1 altered context rejected |
| Z5 | Public-input aliasing (x vs x + r) bypassing replay protection | Canonicity check (< r) in `Verifier.sol` (**bug fixed**: it previously checked < q) and in `parse_statement_v2` | `test_parse_statement_v2_layout` |
| Z6 | Values outside the range assumed by the gadgets (wrap-around mod r) | Range proofs on every ReLU / comparison / rescale | out-of-range test in `test_gadgets` |
| Z7 | Routing manipulation (cheap path although margin < τ) | v1/v2: margin checked against public scores; private: flag computed in-circuit | E4; protocol v2 tests |
| Z8 | Brute-forcing a private low-entropy input | Mandatory random blinding `r` in `C_input` | `misuse` test: refusal without blinding |
| Z9 | Toxic waste of the setup | Phase-2 MPC (`zkml-ceremony`): sound if one participant is honest; PoK + pairing checks per contribution | `test_ceremony` (algebra, tampering); E5 |

## Assumptions and limitations (explicit)

- **Setup phase 1** (powers of tau) is still single-party (`Groth16Prover::setup`).
  The ceremony covers only phase 2 (δ). For production you need a public
  multi-party powers-of-tau, or a universal/transparent proof system.
- **Quantization**: a proof certifies the *integer* model. Drift from the float
  model is measured (trained digits model: 97.3% → 97.1%) but is not proven.
- **Architecture** (layer types and sizes) is public in every mode.
- **Training data and model quality** are not part of the proof.
- **Side channels of the GPU prover** (timing, memory) when the witness is private
  (Phase 3): the prover is not constant-time.
- **Generic Groth16 malleability**: someone can re-randomize a valid proof for the
  same public inputs. Re-randomizing does not change them, so the context still
  prevents use in another session. Uniqueness of the proof itself is not a goal.
- No independent cryptographic audit has been done (Phase 3).

## Outside the model

- Compromise of the prover's machine during private-witness proving.
- Denial of service against the verifier.
- Errors in the semantics of the original (float) model.
