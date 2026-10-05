# PCANI 2026 — proof-carrying adaptive neural inference

## Research question

Can a neural inference system reduce **measured cryptographic proving cost** on easy inputs while staying inside a predeclared accuracy-loss budget, while the verifier still checks that the selected model/path is cryptographically valid?

## Candidate paths and routing

For candidate paths `P1..PK`, ordered by measured standalone prover cost, PCANI calibrates a margin threshold for every non-final path. At inference time it evaluates candidates from cheapest to most expensive and selects the first path whose top-1/top-2 margin clears its threshold; the full path is the fallback.

## Protocol v1 — preserved negative baseline

Protocol v1 required a valid proof for every attempted prefix path. Its optimization objective used cumulative proof-chain cost:

`min E[C_chain(route(x))]`

subject to

`Accuracy(policy) >= Accuracy(full) - epsilon`.

The real T4 run showed that this design is cryptographically valid but inefficient: expected proof cost was 2899.95 ms versus 1884.02 ms for static p160. This negative result is intentionally preserved.

## Protocol v2 — single selected-proof certificate

Protocol v2 uses selected-proof cost:

`min E[C_selected(route(x))]`

subject to the same accuracy constraint.

The prover sends exactly **one** model-specific Groth16 proof for the chosen path. The verifier checks:

1. the policy digest;
2. that the chosen path is permitted by the manifest;
3. the model-specific verification-key SHA-256;
4. native Groth16 proof validity;
5. the public model tag fixed by the circuit;
6. the proof-bound public input/output layout;
7. for a non-final path, `margin >= calibrated_threshold`;
8. the final path as unconditional fallback.

Protocol v2 proves that the selected path is **valid** under the calibrated acceptance rule. It deliberately does **not** prove that every cheaper path would have rejected the input. Route minimality is an optimization property of the prover/policy, not a cryptographic soundness condition.

Implemented in:

- `python/zkml/adaptive.py`
- `python/zkml/pcani_protocol_v2.py`
- `python/pcani_verify_single.py`
- `python/pcani_single_proof_finalize.py`
- `tests/test_pcani_protocol.py`

## Exact-model statement

`zkml-prove --pcani-statement --integer-model --integer-input` builds a model-specific R1CS in which quantized model parameters are **fixed circuit coefficients**, not free witness variables. The verification key is therefore specific to that exact finite-field model.

The statement exposes:

1. pre-softmax output scores;
2. an audit model tag fixed by the circuit;
3. the exact quantized input vector.

The public 64-bit model tag is an audit/debug identifier. Exact model binding comes from the fixed circuit plus the pinned model-specific verification key.

## Privacy scope

- **Input:** public in this research release.
- **Model:** exactly bound, but model hiding is not claimed.
- **Route/confidence:** public to the verifier.

Private input commitments and hidden routing predicates are optional future privacy extensions.

## Real benchmark and measured result

The benchmark uses `sklearn.datasets.load_digits` and exact integer arithmetic with widths `16,48,96,160`.

The completed T4 run measured median prover costs of:

- p16: 1475.75 ms
- p48: 1628.66 ms
- p96: 1781.65 ms
- p160: 1884.02 ms

The held-out routing trace produced 96.11% adaptive accuracy versus 96.94% for p160. Under protocol v2, expected selected-proof cost is 1602.51 ms, a 14.94% reduction relative to p160.

See `GPU_VALIDATION_RESULTS_2026.md` for the evidence boundary.
