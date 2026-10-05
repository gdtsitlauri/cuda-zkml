"""PCANI proof-chain protocol helpers.

Protocol v1 intentionally uses an exact-model Groth16 circuit and a *public*
quantized input vector.  Each attempted path has its own model-specific
verification key.  A valid adaptive route is a prefix of paths P0..Pk where:

* every Groth16 proof verifies;
* every proof exposes the same quantized input;
* every proof exposes the model tag fixed by its circuit;
* the manifest pins each model-specific verification key by SHA-256;
* every cheaper attempted path fails its calibrated confidence threshold; and
* the chosen path clears its threshold, unless it is the full fallback path.

The public-input design is deliberate: it gives exact cross-proof linkage
without pretending that an unaudited in-circuit hash already provides private
input commitments.  A future privacy extension can replace the public vector
with a reviewed field-friendly commitment gadget while retaining the same
proof-chain semantics.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import os
import subprocess
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .artifacts import BN254_SCALAR_FIELD, load_public_inputs


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fr_to_signed(value: int) -> int:
    """Map an Fr standard representative to its symmetric signed integer."""
    value = int(value) % BN254_SCALAR_FIELD
    if value > BN254_SCALAR_FIELD // 2:
        return value - BN254_SCALAR_FIELD
    return value


def top1_margin(values: Sequence[int]) -> int:
    if len(values) < 2:
        raise ValueError("at least two public output scores are required")
    signed = sorted((fr_to_signed(v) for v in values), reverse=True)
    return int(signed[0] - signed[1])


@dataclass(frozen=True)
class ManifestPath:
    name: str
    threshold: Optional[float]
    vk_sha256: str
    model_tag64: str
    model_sha256: str = ""
    standalone_proof_cost_ms: Optional[float] = None

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> "ManifestPath":
        return cls(
            name=str(data["name"]),
            threshold=None if data.get("threshold") is None else float(data["threshold"]),
            vk_sha256=str(data["vk_sha256"]).lower(),
            model_tag64=str(data["model_tag64"]).lower(),
            model_sha256=str(data.get("model_sha256", "")).lower(),
            standalone_proof_cost_ms=(
                None if data.get("standalone_proof_cost_ms") is None
                else float(data["standalone_proof_cost_ms"])
            ),
        )


@dataclass(frozen=True)
class ProtocolManifest:
    policy_digest: str
    confidence_kind: str
    paths: Tuple[ManifestPath, ...]
    protocol_version: int = 1
    statement_mode: str = "pcani-fixed-model/public-input"

    def to_dict(self) -> Dict[str, object]:
        return {
            "protocol_version": self.protocol_version,
            "statement_mode": self.statement_mode,
            "policy_digest": self.policy_digest,
            "confidence_kind": self.confidence_kind,
            "paths": [asdict(p) for p in self.paths],
        }

    def digest(self) -> str:
        canonical = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(canonical).hexdigest()

    def save(self, path: str) -> None:
        payload = self.to_dict()
        payload["manifest_digest"] = self.digest()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")

    @classmethod
    def load(cls, path: str) -> "ProtocolManifest":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        version = int(data.get("protocol_version", 0))
        if version not in (1, 2):
            raise ValueError("unsupported PCANI protocol version")
        obj = cls(
            protocol_version=version,
            statement_mode=str(data.get("statement_mode", "")),
            policy_digest=str(data["policy_digest"]),
            confidence_kind=str(data.get("confidence_kind", "margin")),
            paths=tuple(ManifestPath.from_dict(p) for p in data["paths"]),
        )
        stored = data.get("manifest_digest")
        if stored and str(stored).lower() != obj.digest():
            raise ValueError("manifest digest mismatch")
        if not obj.paths:
            raise ValueError("manifest must contain at least one path")
        if obj.paths[-1].threshold is not None:
            raise ValueError("full fallback path must have threshold=null")
        return obj


@dataclass(frozen=True)
class AttemptArtifact:
    path_name: str
    proof: str
    vk: str
    public_inputs: str
    statement_meta: str

    @classmethod
    def from_dict(cls, data: Mapping[str, object], base_dir: str = "") -> "AttemptArtifact":
        def resolve(v: object) -> str:
            p = str(v)
            return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))
        return cls(
            path_name=str(data["path_name"]),
            proof=resolve(data["proof"]),
            vk=resolve(data["vk"]),
            public_inputs=resolve(data["public_inputs"]),
            statement_meta=resolve(data["statement_meta"]),
        )


@dataclass
class VerificationReport:
    valid: bool
    chosen_path: str
    attempts: int
    input_size: int
    route_margins: Dict[str, int]
    native_verification: Dict[str, bool]
    errors: List[str]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _load_statement(meta_path: str, public_inputs_path: str) -> Tuple[List[int], int, List[int], str]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if meta.get("statement_mode") != "pcani-fixed-model":
        raise ValueError(f"unexpected statement mode in {meta_path}")
    if meta.get("input_visibility") != "public":
        raise ValueError("PCANI protocol v1 requires public input linkage")
    nout = int(meta["public_output_count"])
    tag_idx = int(meta["model_tag_public_index"])
    input_start = int(meta["input_public_start_index"])
    input_size = int(meta["input_size"])
    if tag_idx != nout or input_start != nout + 1:
        raise ValueError("unsupported public-input layout")
    public = load_public_inputs(public_inputs_path)
    expected_len = nout + 1 + input_size
    if len(public) != expected_len:
        raise ValueError(f"public input length {len(public)} != expected {expected_len}")
    outputs = public[:nout]
    model_tag = public[tag_idx]
    linked_input = public[input_start:]
    return outputs, int(model_tag), linked_input, str(meta["model_tag64"]).lower()


def _run_native_verify(verify_cmd: str, attempt: AttemptArtifact) -> Tuple[bool, str]:
    cmd = [
        verify_cmd,
        "--vk", attempt.vk,
        "--proof", attempt.proof,
        "--public-inputs", attempt.public_inputs,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode == 0, text


def verify_route_bundle(bundle_path: str, manifest_path: str, verify_cmd: str) -> VerificationReport:
    manifest = ProtocolManifest.load(manifest_path)
    if manifest.protocol_version != 1:
        raise ValueError("proof-chain verifier requires protocol_version=1 manifest")
    with open(bundle_path, "r", encoding="utf-8") as f:
        bundle = json.load(f)

    errors: List[str] = []
    if int(bundle.get("protocol_version", 0)) != 1:
        errors.append("unsupported bundle protocol version")
    if str(bundle.get("policy_digest", "")) != manifest.policy_digest:
        errors.append("policy digest mismatch")

    chosen_name = str(bundle.get("chosen_path", ""))
    names = [p.name for p in manifest.paths]
    if chosen_name not in names:
        errors.append("chosen path is not in manifest")
        chosen_idx = -1
    else:
        chosen_idx = names.index(chosen_name)

    base_dir = os.path.dirname(os.path.abspath(bundle_path))
    attempts = [AttemptArtifact.from_dict(a, base_dir) for a in bundle.get("attempts", [])]
    if chosen_idx >= 0:
        expected_names = names[: chosen_idx + 1]
        got_names = [a.path_name for a in attempts]
        if got_names != expected_names:
            errors.append(f"attempted paths must be exact prefix {expected_names}, got {got_names}")

    linked_input: Optional[List[int]] = None
    route_margins: Dict[str, int] = {}
    native: Dict[str, bool] = {}

    for idx, attempt in enumerate(attempts):
        if idx >= len(manifest.paths):
            errors.append("bundle contains more attempts than manifest paths")
            break
        spec = manifest.paths[idx]
        if attempt.path_name != spec.name:
            errors.append(f"path order mismatch at index {idx}")
            continue
        for path in [attempt.proof, attempt.vk, attempt.public_inputs, attempt.statement_meta]:
            if not os.path.exists(path):
                errors.append(f"missing artifact: {path}")
        if errors and any(not os.path.exists(p) for p in [attempt.proof, attempt.vk, attempt.public_inputs, attempt.statement_meta]):
            continue

        vk_digest = sha256_file(attempt.vk)
        if vk_digest.lower() != spec.vk_sha256:
            errors.append(f"verification-key digest mismatch for {spec.name}")

        ok, _ = _run_native_verify(verify_cmd, attempt)
        native[spec.name] = ok
        if not ok:
            errors.append(f"native Groth16 verification failed for {spec.name}")

        try:
            outputs, model_tag, this_input, meta_tag = _load_statement(
                attempt.statement_meta, attempt.public_inputs
            )
        except Exception as exc:
            errors.append(f"statement parse failed for {spec.name}: {exc}")
            continue

        tag_expected = int(spec.model_tag64, 16)
        if model_tag != tag_expected:
            errors.append(f"public model tag mismatch for {spec.name}")
        if int(meta_tag, 16) != tag_expected:
            errors.append(f"statement metadata tag mismatch for {spec.name}")

        if linked_input is None:
            linked_input = list(this_input)
        elif list(this_input) != linked_input:
            errors.append(f"cross-proof input linkage failed at {spec.name}")

        if manifest.confidence_kind != "margin":
            errors.append("protocol v1 currently supports confidence_kind=margin only")
            continue
        margin = top1_margin(outputs)
        route_margins[spec.name] = margin

        # All paths before the chosen path must fail their threshold.  The
        # chosen non-final path must clear its threshold.  The final path is an
        # unconditional fallback.
        if chosen_idx >= 0:
            if idx < chosen_idx:
                if spec.threshold is None or margin >= float(spec.threshold):
                    errors.append(f"route rule violated: {spec.name} should have rejected")
            elif idx == chosen_idx and idx < len(manifest.paths) - 1:
                if spec.threshold is None or margin < float(spec.threshold):
                    errors.append(f"route rule violated: {spec.name} should have accepted")

    input_size = len(linked_input or [])
    return VerificationReport(
        valid=not errors,
        chosen_path=chosen_name,
        attempts=len(attempts),
        input_size=input_size,
        route_margins=route_margins,
        native_verification=native,
        errors=errors,
    )


def make_bundle(
    output_path: str,
    *,
    policy_digest: str,
    chosen_path: str,
    attempts: Sequence[Mapping[str, str]],
) -> None:
    payload = {
        "protocol_version": 1,
        "policy_digest": policy_digest,
        "chosen_path": chosen_path,
        "attempts": list(attempts),
    }
    base = os.path.dirname(os.path.abspath(output_path))
    # Keep bundle relocatable by storing paths relative to the bundle when possible.
    for item in payload["attempts"]:
        for key in ("proof", "vk", "public_inputs", "statement_meta"):
            if key in item:
                try:
                    item[key] = os.path.relpath(os.path.abspath(item[key]), base)
                except Exception:
                    pass
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


__all__ = [
    "AttemptArtifact",
    "ManifestPath",
    "ProtocolManifest",
    "VerificationReport",
    "fr_to_signed",
    "make_bundle",
    "sha256_file",
    "top1_margin",
    "verify_route_bundle",
]
