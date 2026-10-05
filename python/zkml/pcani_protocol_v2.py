"""PCANI protocol v2: single selected-path proof certificates.

Protocol v1 verified an exact prefix of proofs P0..Pk. That makes routing fully
replayable but pays cumulative proving cost. Protocol v2 separates *soundness*
from *minimality*: the prover may select any permitted path, but the verifier
accepts it only when

  1. the model-specific Groth16 proof verifies,
  2. the verification key and model tag are pinned by the manifest,
  3. the proof-bound public input is present in the statement, and
  4. the selected path's proof-bound margin clears its calibrated threshold
     (the final path remains an unconditional fallback).

The verifier does not require proofs that cheaper paths would have rejected.
Therefore protocol v2 proves validity of the selected route, not that it was the
cheapest valid route. A rational/benchmark prover can still evaluate paths from
cheap to expensive and emit only the first valid certificate. This preserves
cryptographic correctness while avoiding the cumulative proof-chain overhead
measured in the 2026 T4 validation run.

Anti-replay (roadmap 1.1): statements produced with ``zkml-prove --statement-v2
--context`` carry a public context H(session || nonce || time || policy) bound in
the R1CS. A verifier that issues a fresh challenge passes ``expected_context``;
a proof made for another session, or replayed (``seen_contexts``), is rejected.
With private outputs the route decision is the proof-bound in-circuit flag.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import os
from typing import Dict, List, Mapping, MutableSet, Optional

from .pcani_protocol import (
    AttemptArtifact,
    ProtocolManifest,
    _load_statement,
    _run_native_verify,
    sha256_file,
    top1_margin,
)
from .statement_v2 import load_statement_v2


@dataclass
class SingleProofVerificationReport:
    valid: bool
    chosen_path: str
    input_size: int
    route_margin: int | None
    native_verification: bool
    errors: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def verify_single_proof_certificate(
    certificate_path: str,
    manifest_path: str,
    verify_cmd: str,
    *,
    expected_context: Optional[int] = None,
    seen_contexts: Optional[MutableSet[int]] = None,
) -> SingleProofVerificationReport:
    manifest = ProtocolManifest.load(manifest_path)
    if manifest.protocol_version != 2:
        raise ValueError("single-proof verifier requires protocol_version=2 manifest")
    with open(certificate_path, "r", encoding="utf-8") as f:
        cert = json.load(f)

    errors: List[str] = []
    if int(cert.get("protocol_version", 0)) != 2:
        errors.append("unsupported certificate protocol version")
    if str(cert.get("policy_digest", "")) != manifest.policy_digest:
        errors.append("policy digest mismatch")

    chosen = str(cert.get("chosen_path", ""))
    by_name = {p.name: p for p in manifest.paths}
    if chosen not in by_name:
        errors.append("chosen path is not in manifest")
        return SingleProofVerificationReport(False, chosen, 0, None, False, errors)
    spec = by_name[chosen]

    base_dir = os.path.dirname(os.path.abspath(certificate_path))
    try:
        attempt = AttemptArtifact.from_dict(cert["attempt"], base_dir)
    except Exception as exc:
        errors.append(f"invalid attempt: {exc}")
        return SingleProofVerificationReport(False, chosen, 0, None, False, errors)

    if attempt.path_name != chosen:
        errors.append("attempt path does not match chosen path")

    missing = [p for p in [attempt.proof, attempt.vk, attempt.public_inputs, attempt.statement_meta] if not os.path.exists(p)]
    for p in missing:
        errors.append(f"missing artifact: {p}")
    if missing:
        return SingleProofVerificationReport(False, chosen, 0, None, False, errors)

    if sha256_file(attempt.vk).lower() != spec.vk_sha256:
        errors.append(f"verification-key digest mismatch for {chosen}")

    native_ok, _ = _run_native_verify(verify_cmd, attempt)
    if not native_ok:
        errors.append(f"native Groth16 verification failed for {chosen}")

    input_size = 0
    margin = None
    with open(attempt.statement_meta, "r", encoding="utf-8") as f:
        is_v2 = json.load(f).get("statement_mode") == "pcani-v2"
    if is_v2:
        return _verify_v2_statement(manifest, spec, chosen, attempt, native_ok, errors,
                                    expected_context, seen_contexts)
    if expected_context is not None:
        errors.append("statement is not bound to a context (v1 statement; replayable)")

    try:
        outputs, model_tag, linked_input, meta_tag = _load_statement(
            attempt.statement_meta, attempt.public_inputs
        )
        input_size = len(linked_input)
        expected_tag = int(spec.model_tag64, 16)
        if model_tag != expected_tag:
            errors.append(f"public model tag mismatch for {chosen}")
        if int(meta_tag, 16) != expected_tag:
            errors.append(f"statement metadata tag mismatch for {chosen}")

        if manifest.confidence_kind != "margin":
            errors.append("protocol v2 currently supports confidence_kind=margin only")
        else:
            margin = top1_margin(outputs)
            is_final = chosen == manifest.paths[-1].name
            if not is_final:
                if spec.threshold is None:
                    errors.append(f"non-final path {chosen} has no acceptance threshold")
                elif margin < float(spec.threshold):
                    errors.append(
                        f"selected path {chosen} failed acceptance threshold: "
                        f"margin={margin} < threshold={spec.threshold}"
                    )
    except Exception as exc:
        errors.append(f"statement parse failed for {chosen}: {exc}")

    return SingleProofVerificationReport(
        valid=not errors,
        chosen_path=chosen,
        input_size=input_size,
        route_margin=margin,
        native_verification=native_ok,
        errors=errors,
    )


def _verify_v2_statement(manifest, spec, chosen, attempt, native_ok, errors,
                         expected_context, seen_contexts) -> SingleProofVerificationReport:
    input_size, margin = 0, None
    try:
        st = load_statement_v2(attempt.statement_meta, attempt.public_inputs)
        input_size = int(st.meta["input_size"])
        if st.model_tag != int(spec.model_tag64, 16):
            errors.append(f"public model tag mismatch for {chosen}")
        if expected_context is not None:
            if st.context is None:
                errors.append("statement is not bound to a context")
            elif st.context != expected_context:
                errors.append("context mismatch (proof made for another session)")
        if st.context is not None and seen_contexts is not None:
            if st.context in seen_contexts:
                errors.append("context already used (replay)")
            elif not errors:
                seen_contexts.add(st.context)
        is_final = chosen == manifest.paths[-1].name
        if not is_final:
            if st.route:
                # private or public outputs: the in-circuit route flag is authoritative
                if any(flag != 1 for flag, _ in st.route):
                    errors.append(f"in-circuit route flag rejects path {chosen}")
                if spec.threshold is not None and int(st.meta["route_threshold"]) < float(spec.threshold):
                    errors.append("statement route threshold below the manifest threshold")
            elif st.outputs:
                if spec.threshold is None:
                    errors.append(f"non-final path {chosen} has no acceptance threshold")
                else:
                    # outputs are already signed; every sample of a batch must clear the threshold
                    margin = min(sorted(out, reverse=True)[0] - sorted(out, reverse=True)[1]
                                 for out in st.outputs)
                    if margin < float(spec.threshold):
                        errors.append(f"selected path {chosen} failed acceptance threshold: "
                                      f"margin={margin} < threshold={spec.threshold}")
            else:
                errors.append("statement exposes neither outputs nor a route flag")
    except Exception as exc:
        errors.append(f"statement parse failed for {chosen}: {exc}")
    return SingleProofVerificationReport(valid=not errors, chosen_path=chosen, input_size=input_size,
                                         route_margin=margin, native_verification=native_ok, errors=errors)


def make_single_proof_certificate(
    output_path: str,
    *,
    policy_digest: str,
    chosen_path: str,
    attempt: Mapping[str, str],
) -> None:
    payload = {
        "protocol_version": 2,
        "policy_digest": str(policy_digest),
        "chosen_path": str(chosen_path),
        "attempt": dict(attempt),
    }
    base = os.path.dirname(os.path.abspath(output_path))
    for key in ("proof", "vk", "public_inputs", "statement_meta"):
        if key in payload["attempt"]:
            try:
                payload["attempt"][key] = os.path.relpath(
                    os.path.abspath(payload["attempt"][key]), base
                )
            except Exception:
                pass
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


__all__ = [
    "SingleProofVerificationReport",
    "verify_single_proof_certificate",
    "make_single_proof_certificate",
]
