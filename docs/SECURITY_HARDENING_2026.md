# Security Hardening Notes — 2026

## 1. Toxic-waste handling

The original proving-key format persisted setup scalars and scalar-domain query exponents. That is unacceptable for a production Groth16 trusted-setup boundary.

The hardened branch changes the default behavior:

- setup trapdoor values exist only as local variables;
- returned proving keys hold group-encoded queries;
- serialized PK format is `ZKMLPK3` / version 3;
- v3 serializes group elements only;
- scalar query vectors and debug trapdoor state cause `ProvingKey::save()` to fail closed;
- legacy v2 files are rejected and must be regenerated.

`ZKML_UNSAFE_KEEP_TRAPDOOR=1` is retained solely for local diagnostics. A PK created in that mode cannot be serialized by the hardened writer.

## 2. Groth16 C element

The prover uses the **unblinded** G1 analogue

`B1 = beta + sum_i w_i B_i`,

while `A` already includes `r*delta`. Under this convention the pairing-consistent Groth16 construction is

`C = L + H + s*A + r*B1`.

The `s*A` term already contains the single `r*s*delta` contribution. An explicit additional `-r*s*delta` therefore over-corrects and makes valid proofs fail verification. This was caught by the first NVIDIA/Colab end-to-end validation and corrected in the GPU-validation patch.

## 3. Aggregation

A random linear combination of Groth16 `(A,B,C)` tuples is not, by itself, a sound general aggregation protocol. The old `aggregate_proofs` API is retained only for source compatibility and returns an invalid proof with a clear warning.

Until a formally specified protocol is integrated, multiple proofs must be verified independently or through a separate audited aggregation scheme.

## 4. Model identity / authenticity

The current R1CS treats private weights as witness variables. Therefore the proof establishes consistency with *some* satisfying private weights, not with a public commitment to a named proprietary model.

A future exact-model mode must use one of the following defensible constructions:

- a ZK-friendly cryptographic commitment/hash of private weights constrained inside the proof,
- a polynomial/vector commitment with efficient linkage to the weight witness,
- or a circuit-specific fixed-model construction with an explicit privacy/security analysis.

A plain SHA-256 metadata file outside the circuit is useful for artifact bookkeeping but **does not solve this cryptographic binding problem**.

## 5. Remaining security validation

Before any production/security claim:

- rerun CUDA tests on NVIDIA hardware;
- use independent test vectors for field, curve, pairing, and Groth16;
- fuzz malformed proof/VK/PK deserialization;
- add subgroup/canonical-encoding checks where required;
- obtain external cryptographic review;
- replace simulated trusted setup with a defensible ceremony or universal/updatable SRS design.
