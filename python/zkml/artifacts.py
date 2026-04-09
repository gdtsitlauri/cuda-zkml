"""
Artifact parsing and Solidity export helpers for CUDA-zkML.

These helpers parse the repository's binary proof, verification key, and
public-input formats and convert them into Solidity/Ethers-friendly JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
import json
import os
import struct
from typing import Any, BinaryIO, Dict, List, Sequence, Tuple


LIMBS_PER_FIELD_ELEMENT = 4
BYTES_PER_LIMB = 8
BYTES_PER_FIELD_ELEMENT = LIMBS_PER_FIELD_ELEMENT * BYTES_PER_LIMB
BN254_PRIME = 21888242871839275222246405745257275088696311157297823662689037894645226208583


def _read_exact(stream: BinaryIO, size: int, label: str) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError(f"Truncated {label}: expected {size} bytes, got {len(data)}")
    return data


def _read_i32(stream: BinaryIO, label: str) -> int:
    return struct.unpack("<i", _read_exact(stream, 4, label))[0]


def _read_limbs(stream: BinaryIO, label: str) -> Tuple[int, int, int, int]:
    return struct.unpack("<4Q", _read_exact(stream, BYTES_PER_FIELD_ELEMENT, label))


def limbs_to_int(limbs: Sequence[int]) -> int:
    value = 0
    for i, limb in enumerate(limbs):
        value |= int(limb) << (64 * i)
    return value


def _read_fp(stream: BinaryIO, label: str) -> int:
    return limbs_to_int(_read_limbs(stream, label))


def _read_g1(stream: BinaryIO, label: str) -> List[int]:
    return [
        _read_fp(stream, f"{label}.x"),
        _read_fp(stream, f"{label}.y"),
    ]


def _read_g2(stream: BinaryIO, label: str) -> List[List[int]]:
    return [
        [
            _read_fp(stream, f"{label}.x.c0"),
            _read_fp(stream, f"{label}.x.c1"),
        ],
        [
            _read_fp(stream, f"{label}.y.c0"),
            _read_fp(stream, f"{label}.y.c1"),
        ],
    ]


def _stringify_uint256_tree(value: Any) -> Any:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return [_stringify_uint256_tree(item) for item in value]
    if isinstance(value, tuple):
        return [_stringify_uint256_tree(item) for item in value]
    if isinstance(value, dict):
        return {key: _stringify_uint256_tree(item) for key, item in value.items()}
    return value


@dataclass
class ProofArtifact:
    a: List[int]
    b: List[List[int]]
    c: List[int]

    @classmethod
    def from_bytes(cls, data: bytes) -> "ProofArtifact":
        if len(data) != 256:
            raise ValueError(f"Proof must be exactly 256 bytes, got {len(data)}")
        stream = BytesIO(data)
        artifact = cls(
            a=_read_g1(stream, "proof.A"),
            b=_read_g2(stream, "proof.B"),
            c=_read_g1(stream, "proof.C"),
        )
        trailing = stream.read()
        if trailing:
            raise ValueError(f"Unexpected trailing bytes in proof: {len(trailing)}")
        return artifact

    @classmethod
    def from_file(cls, path: str) -> "ProofArtifact":
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

    def to_python_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "c": self.c}

    def to_solidity_dict(self) -> Dict[str, Any]:
        return _stringify_uint256_tree(self.to_python_dict())


@dataclass
class VerificationKeyArtifact:
    num_public: int
    alpha: List[int]
    beta: List[List[int]]
    gamma: List[List[int]]
    delta: List[List[int]]
    ic: List[List[int]]

    @classmethod
    def from_bytes(cls, data: bytes) -> "VerificationKeyArtifact":
        stream = BytesIO(data)
        num_public = _read_i32(stream, "vk.num_public")
        alpha = _read_g1(stream, "vk.alpha")
        beta = _read_g2(stream, "vk.beta")
        gamma = _read_g2(stream, "vk.gamma")
        delta = _read_g2(stream, "vk.delta")
        num_ic = _read_i32(stream, "vk.num_ic")
        if num_ic < 0:
            raise ValueError(f"Invalid IC count: {num_ic}")
        ic = [_read_g1(stream, f"vk.ic[{i}]") for i in range(num_ic)]
        trailing = stream.read()
        if trailing:
            raise ValueError(f"Unexpected trailing bytes in VK: {len(trailing)}")
        return cls(
            num_public=num_public,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            ic=ic,
        )

    @classmethod
    def from_file(cls, path: str) -> "VerificationKeyArtifact":
        with open(path, "rb") as f:
            return cls.from_bytes(f.read())

    def to_python_dict(self) -> Dict[str, Any]:
        return {
            "numPublic": self.num_public,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "delta": self.delta,
            "ic": self.ic,
        }

    def to_solidity_dict(self) -> Dict[str, Any]:
        return _stringify_uint256_tree(self.to_python_dict())


def load_public_inputs(path: str) -> List[int]:
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("Public inputs JSON must be a list")
        return [int(value) % BN254_PRIME for value in data]

    with open(path, "rb") as f:
        n = _read_i32(f, "public_inputs.count")
        if n < 0:
            raise ValueError(f"Invalid public input count: {n}")
        return [limbs_to_int(_read_limbs(f, f"public_inputs[{i}]")) for i in range(n)]


def _g1_is_on_evm_curve(point: Sequence[int]) -> bool:
    if len(point) != 2:
        return False
    if int(point[0]) == 0 and int(point[1]) == 0:
        return True

    try:
        from py_ecc.bn128 import FQ, b, is_on_curve
    except ImportError as exc:
        raise ImportError("pip install py-ecc to check EVM BN254 compatibility") from exc

    return bool(is_on_curve((FQ(int(point[0])), FQ(int(point[1]))), b))


def _g2_is_on_evm_curve(point: Sequence[Sequence[int]]) -> bool:
    if len(point) != 2 or len(point[0]) != 2 or len(point[1]) != 2:
        return False
    if all(int(coord) == 0 for row in point for coord in row):
        return True

    try:
        from py_ecc.bn128 import FQ2, b2, is_on_curve
    except ImportError as exc:
        raise ImportError("pip install py-ecc to check EVM BN254 compatibility") from exc

    x = FQ2([int(point[0][0]), int(point[0][1])])
    y = FQ2([int(point[1][0]), int(point[1][1])])
    return bool(is_on_curve((x, y), b2))


def evm_curve_compatibility_report(
    vk: VerificationKeyArtifact,
    proof: ProofArtifact,
) -> Dict[str, Any]:
    vk_ic_checks = [_g1_is_on_evm_curve(point) for point in vk.ic]
    report = {
        "proof": {
            "A": _g1_is_on_evm_curve(proof.a),
            "B": _g2_is_on_evm_curve(proof.b),
            "C": _g1_is_on_evm_curve(proof.c),
        },
        "vk": {
            "alpha": _g1_is_on_evm_curve(vk.alpha),
            "beta": _g2_is_on_evm_curve(vk.beta),
            "gamma": _g2_is_on_evm_curve(vk.gamma),
            "delta": _g2_is_on_evm_curve(vk.delta),
            "ic": vk_ic_checks,
            "ic_all": all(vk_ic_checks),
        },
    }
    report["all_points_evm_compatible"] = bool(
        report["proof"]["A"]
        and report["proof"]["B"]
        and report["proof"]["C"]
        and report["vk"]["alpha"]
        and report["vk"]["beta"]
        and report["vk"]["gamma"]
        and report["vk"]["delta"]
        and report["vk"]["ic_all"]
    )
    return report


def parse_proof_bytes(data: bytes) -> Dict[str, Any]:
    return ProofArtifact.from_bytes(data).to_python_dict()


def parse_verification_key_bytes(data: bytes) -> Dict[str, Any]:
    return VerificationKeyArtifact.from_bytes(data).to_python_dict()


def _write_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def build_solidity_bundle(
    vk: VerificationKeyArtifact,
    proof: ProofArtifact,
    public_inputs: Sequence[int],
) -> Dict[str, Any]:
    solidity_public_inputs = [str(int(value) % BN254_PRIME) for value in public_inputs]
    return {
        "vk": vk.to_solidity_dict(),
        "proof": proof.to_solidity_dict(),
        "publicInputs": solidity_public_inputs,
        "setVerifyingKeyArgs": {
            "alpha": _stringify_uint256_tree(vk.alpha),
            "beta": _stringify_uint256_tree(vk.beta),
            "gamma": _stringify_uint256_tree(vk.gamma),
            "delta": _stringify_uint256_tree(vk.delta),
            "ic": _stringify_uint256_tree(vk.ic),
        },
        "verifyProofArgs": {
            "proof": proof.to_solidity_dict(),
            "publicInputs": solidity_public_inputs,
        },
    }


def export_solidity_bundle(
    vk_path: str,
    proof_path: str,
    public_inputs_path: str,
    output_dir: str,
) -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)

    vk = VerificationKeyArtifact.from_file(vk_path)
    proof = ProofArtifact.from_file(proof_path)
    public_inputs = load_public_inputs(public_inputs_path)
    bundle = build_solidity_bundle(vk, proof, public_inputs)

    _write_json(os.path.join(output_dir, "vk.solidity.json"), bundle["vk"])
    _write_json(os.path.join(output_dir, "proof.solidity.json"), bundle["proof"])
    _write_json(
        os.path.join(output_dir, "public_inputs.solidity.json"),
        {"publicInputs": bundle["publicInputs"]},
    )
    _write_json(os.path.join(output_dir, "calldata.solidity.json"), bundle)

    verifier_call = (
        "await verifier.setVerifyingKey(\n"
        f"  {json.dumps(bundle['setVerifyingKeyArgs']['alpha'])},\n"
        f"  {json.dumps(bundle['setVerifyingKeyArgs']['beta'])},\n"
        f"  {json.dumps(bundle['setVerifyingKeyArgs']['gamma'])},\n"
        f"  {json.dumps(bundle['setVerifyingKeyArgs']['delta'])},\n"
        f"  {json.dumps(bundle['setVerifyingKeyArgs']['ic'])}\n"
        ");\n\n"
        "const proof = "
        f"{json.dumps(bundle['verifyProofArgs']['proof'], indent=2)};\n"
        "const publicInputs = "
        f"{json.dumps(bundle['verifyProofArgs']['publicInputs'], indent=2)};\n"
        "const valid = await verifier.verifyProof(proof, publicInputs);\n"
    )

    with open(os.path.join(output_dir, "verifier_call.txt"), "w", encoding="utf-8") as f:
        f.write(verifier_call)

    return bundle


__all__ = [
    "BN254_PRIME",
    "ProofArtifact",
    "VerificationKeyArtifact",
    "evm_curve_compatibility_report",
    "build_solidity_bundle",
    "export_solidity_bundle",
    "limbs_to_int",
    "load_public_inputs",
    "parse_proof_bytes",
    "parse_verification_key_bytes",
]
