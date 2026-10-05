import hashlib
import json
import os
import stat
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from zkml.pcani_protocol import (
    ManifestPath,
    ProtocolManifest,
    make_bundle,
    verify_route_bundle,
)

R = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def _write_public(path, vals):
    with open(path, "wb") as f:
        f.write(struct.pack("<i", len(vals)))
        for v in vals:
            v %= R
            limbs = [(v >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]
            f.write(struct.pack("<4Q", *limbs))


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _meta(path, tag, nout=3, input_size=2):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "statement_version": 1,
                "statement_mode": "pcani-fixed-model",
                "model_binding": "fixed-quantized-parameters-in-r1cs",
                "model_tag64": hex(tag),
                "input_visibility": "public",
                "public_output_count": nout,
                "model_tag_public_index": nout,
                "input_public_start_index": nout + 1,
                "input_size": input_size,
                "r1cs_constraints": 10,
                "r1cs_variables": 20,
            },
            f,
        )


def _fake_verifier(path):
    with open(path, "w", encoding="utf-8") as f:
        f.write("#!/bin/sh\nexit 0\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


def test_pcani_route_bundle_verifies_exact_prefix_and_input_linkage(tmp_path):
    verify = tmp_path / "verify.sh"
    _fake_verifier(verify)

    paths = []
    attempts = []
    inp = [5, 7]
    # cheap margin=2 -> reject threshold 5; mid margin=9 -> accept threshold 8.
    outputs = [[10, 8, 1], [20, 11, 0]]
    tags = [0x1111, 0x2222]
    names = ["cheap", "mid"]
    thresholds = [5.0, 8.0]
    for i, name in enumerate(names):
        vk = tmp_path / f"{name}.vk"
        proof = tmp_path / f"{name}.proof"
        pi = tmp_path / f"{name}.pi"
        meta = tmp_path / f"{name}.json"
        vk.write_bytes((name + "-vk").encode())
        proof.write_bytes(b"proof")
        _write_public(pi, outputs[i] + [tags[i]] + inp)
        _meta(meta, tags[i])
        paths.append(
            ManifestPath(
                name=name,
                threshold=thresholds[i],
                vk_sha256=_sha(vk),
                model_tag64=hex(tags[i]),
            )
        )
        attempts.append(
            {
                "path_name": name,
                "proof": str(proof),
                "vk": str(vk),
                "public_inputs": str(pi),
                "statement_meta": str(meta),
            }
        )

    # full fallback exists in manifest but is not attempted because mid accepts.
    vk = tmp_path / "full.vk"
    vk.write_bytes(b"full-vk")
    paths.append(ManifestPath("full", None, _sha(vk), "0x3333"))

    manifest = ProtocolManifest("policy-1", "margin", tuple(paths))
    manifest_path = tmp_path / "manifest.json"
    manifest.save(manifest_path)
    bundle_path = tmp_path / "bundle.json"
    make_bundle(
        bundle_path,
        policy_digest="policy-1",
        chosen_path="mid",
        attempts=attempts,
    )

    report = verify_route_bundle(str(bundle_path), str(manifest_path), str(verify))
    assert report.valid, report.errors
    assert report.chosen_path == "mid"
    assert report.route_margins == {"cheap": 2, "mid": 9}


def test_pcani_rejects_cross_proof_input_substitution(tmp_path):
    verify = tmp_path / "verify.sh"
    _fake_verifier(verify)
    paths = []
    attempts = []
    for i, (name, tag, inp) in enumerate([
        ("cheap", 1, [5, 7]),
        ("full", 2, [5, 8]),
    ]):
        vk = tmp_path / f"{name}.vk"; vk.write_bytes(name.encode())
        proof = tmp_path / f"{name}.proof"; proof.write_bytes(b"p")
        pi = tmp_path / f"{name}.pi"; _write_public(pi, [10, 10, 1, tag] + inp)
        meta = tmp_path / f"{name}.json"; _meta(meta, tag)
        paths.append(ManifestPath(name, 1.0 if i == 0 else None, _sha(vk), hex(tag)))
        attempts.append({"path_name": name, "proof": str(proof), "vk": str(vk), "public_inputs": str(pi), "statement_meta": str(meta)})
    manifest = ProtocolManifest("p", "margin", tuple(paths))
    mp = tmp_path / "m.json"; manifest.save(mp)
    bp = tmp_path / "b.json"; make_bundle(bp, policy_digest="p", chosen_path="full", attempts=attempts)
    report = verify_route_bundle(str(bp), str(mp), str(verify))
    assert not report.valid
    assert any("input linkage" in e for e in report.errors)


def test_pcani_v2_single_proof_certificate_accepts_selected_path(tmp_path):
    from zkml.pcani_protocol_v2 import make_single_proof_certificate, verify_single_proof_certificate

    verify = tmp_path / "verify.sh"
    _fake_verifier(verify)
    inp = [5, 7]
    paths = []
    artifacts = {}
    for name, tag, threshold, outputs in [
        ("cheap", 0x1111, 5.0, [10, 8, 1]),
        ("mid", 0x2222, 8.0, [20, 11, 0]),
        ("full", 0x3333, None, [20, 11, 0]),
    ]:
        vk = tmp_path / f"{name}.vk"; vk.write_bytes((name + "-vk").encode())
        proof = tmp_path / f"{name}.proof"; proof.write_bytes(b"proof")
        pi = tmp_path / f"{name}.pi"; _write_public(pi, outputs + [tag] + inp)
        meta = tmp_path / f"{name}.json"; _meta(meta, tag)
        paths.append(ManifestPath(name, threshold, _sha(vk), hex(tag)))
        artifacts[name] = {
            "path_name": name, "proof": str(proof), "vk": str(vk),
            "public_inputs": str(pi), "statement_meta": str(meta),
        }

    manifest = ProtocolManifest("policy-v2", "margin", tuple(paths), protocol_version=2)
    mp = tmp_path / "manifest.json"; manifest.save(mp)
    cp = tmp_path / "cert.json"
    make_single_proof_certificate(cp, policy_digest="policy-v2", chosen_path="mid", attempt=artifacts["mid"])
    report = verify_single_proof_certificate(str(cp), str(mp), str(verify))
    assert report.valid, report.errors
    assert report.chosen_path == "mid"
    assert report.route_margin == 9
    assert report.native_verification


def test_pcani_v2_single_proof_certificate_rejects_below_threshold(tmp_path):
    from zkml.pcani_protocol_v2 import make_single_proof_certificate, verify_single_proof_certificate

    verify = tmp_path / "verify.sh"
    _fake_verifier(verify)
    vk = tmp_path / "cheap.vk"; vk.write_bytes(b"cheap-vk")
    proof = tmp_path / "cheap.proof"; proof.write_bytes(b"proof")
    pi = tmp_path / "cheap.pi"; _write_public(pi, [10, 8, 1, 0x1111, 5, 7])
    meta = tmp_path / "cheap.json"; _meta(meta, 0x1111)
    full_vk = tmp_path / "full.vk"; full_vk.write_bytes(b"full-vk")
    manifest = ProtocolManifest(
        "policy-v2", "margin",
        (
            ManifestPath("cheap", 5.0, _sha(vk), "0x1111"),
            ManifestPath("full", None, _sha(full_vk), "0x2222"),
        ),
        protocol_version=2,
    )
    mp = tmp_path / "manifest.json"; manifest.save(mp)
    cp = tmp_path / "cert.json"
    make_single_proof_certificate(
        cp, policy_digest="policy-v2", chosen_path="cheap",
        attempt={"path_name": "cheap", "proof": str(proof), "vk": str(vk), "public_inputs": str(pi), "statement_meta": str(meta)},
    )
    report = verify_single_proof_certificate(str(cp), str(mp), str(verify))
    assert not report.valid
    assert any("failed acceptance threshold" in e for e in report.errors)
