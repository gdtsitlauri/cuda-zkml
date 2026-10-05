"""Python-side tests for PCANI statement v2 (roadmap 1.1, 2.1-2.4).

Covers: Poseidon reference vector, commitments, context derivation, v2 statement
parsing, exact integer reference inference (RELU_EXACT / CONV2D), routing
semantics, and the protocol-v2 verifier's context / replay / route checks
(native Groth16 verification is stubbed here; it is exercised end-to-end on GPU
in notebooks/PCANI_Colab_Phase12.ipynb).
"""
import hashlib
import json
import os
import struct
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from zkml import poseidon  # noqa: E402
from zkml import pcani_protocol_v2 as v2proto  # noqa: E402
from zkml.pcani_protocol import ManifestPath, ProtocolManifest  # noqa: E402
from zkml.statement_v2 import (  # noqa: E402
    R, compute_context, input_commitment, model_commitment, parse_statement_v2,
    reference_scores, route_decision,
)


def test_poseidon_matches_reference_and_circomlib():
    assert poseidon.permute(poseidon.TEST_IN) == poseidon.TEST_OUT
    assert poseidon.hash2(1, 2) == 7853200120776062878684798364095072458815029376092732009249414926327459813530


def test_commitments_match_cpp_vectors():
    # values asserted by tests/host/test_gadgets.cpp on the C++ side
    assert poseidon.commit([1, 2, 3, 4, 5], poseidon.DOMAIN_MODEL) == \
        3508929059074620887801063817879916348866360521201526977122764380756820441871
    assert input_commitment([7], -3) == poseidon.commit([7, -3 % R], poseidon.DOMAIN_INPUT)
    assert model_commitment([[1, 2], [3, 4, 5]]) == poseidon.commit([1, 2, 3, 4, 5], poseidon.DOMAIN_MODEL)
    assert input_commitment([1, 2], 5) != input_commitment([1, 2], 6)   # blinding hides the input


def test_context_is_deterministic_bounded_and_session_specific():
    c1 = compute_context("session-A", b"nonce1", 1700000000, "ab" * 32)
    assert c1 == compute_context("session-A", b"nonce1", 1700000000, "ab" * 32)
    assert 0 <= c1 < (1 << 253) < R
    assert c1 != compute_context("session-B", b"nonce1", 1700000000, "ab" * 32)
    assert c1 != compute_context("session-A", b"nonce2", 1700000000, "ab" * 32)
    assert c1 != compute_context("session-A", b"nonce1", 1700000001, "ab" * 32)
    # length-prefixed encoding: no ambiguity between field boundaries
    assert compute_context("ab", b"c", 1, "00") != compute_context("a", b"bc", 1, "00")


def test_reference_relu_exact_conv_and_route():
    layers = [
        {"type": "conv2d", "in_features": 9, "out_features": 4, "in_channels": 1, "in_height": 3,
         "in_width": 3, "out_channels": 1, "kernel_size": 2, "stride": 1, "padding": 0},
        {"type": "relu_exact", "in_features": 4, "out_features": 4, "bits": 16, "shift": 1},
        {"type": "linear", "in_features": 4, "out_features": 2},
    ]
    params = [1, -1, 2, 0, 3,   1, 0, 0, 1, 0, 1, 1, 0, -1, 2]   # conv w(4)+b(1), linear w(8)+b(2)
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # conv: y = b + w00*x[i][j] - w01*x[i][j+1] + 2*x[i+1][j]
    conv = [3 + x[r * 3 + c] - x[r * 3 + c + 1] + 2 * x[(r + 1) * 3 + c] for r in range(2) for c in range(2)]
    hidden = [max(v, 0) >> 1 for v in conv]
    expect = [hidden[0] + hidden[3] - 1, hidden[1] + hidden[2] + 2]
    assert reference_scores(layers, params, x) == expect
    with pytest.raises(ValueError):
        reference_scores([layers[1]], [], [1 << 15])         # outside the 16-bit range
    assert route_decision([5, 9, 9, 1], 1) == (0, 1)          # tie -> lowest index, margin 0
    assert route_decision([5, 20, 9, 1], 10) == (1, 1)


def _meta_v2(**kw):
    meta = {"statement_version": 2, "statement_mode": "pcani-v2", "model_tag64": "0x1234",
            "input_visibility": "public", "output_visibility": "public", "batch": 1, "input_size": 2,
            "output_size": 3, "outputs_start_index": 0, "model_tag_public_index": 3,
            "context_public_index": 4, "model_commitment_index": -1, "inputs_start_index": 5,
            "input_slot_size": 2, "route_start_index": -1, "route_threshold": 0, "num_public": 7,
            "r1cs_constraints": 1, "r1cs_variables": 1}
    meta.update(kw)
    return meta


def test_parse_statement_v2_layout():
    st = parse_statement_v2(_meta_v2(), [10, R - 3, 1, 0x1234, 99, 5, R - 7])
    assert st.outputs == [[10, -3, 1]] and st.context == 99 and st.inputs == [[5, -7]]
    with pytest.raises(ValueError):
        parse_statement_v2(_meta_v2(), [10, 1, 1, 0x9999, 99, 5, 7])   # tag mismatch
    with pytest.raises(ValueError):
        parse_statement_v2(_meta_v2(), [10, 1, 1, 0x1234, 99 + R, 5, 7])   # aliased context x + r


def _write_public_json(path, vals):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([str(v % R) for v in vals], f)


def _setup_certificate(tmp_path, context, outputs=(30, 10, 1)):
    vk = tmp_path / "p.vk"
    vk.write_bytes(b"vk")
    proof = tmp_path / "p.proof"
    proof.write_bytes(b"proof")
    pi = tmp_path / "p.json"
    _write_public_json(pi, list(outputs) + [0x1234, context, 5, 7])
    meta = tmp_path / "p.meta.json"
    meta.write_text(json.dumps(_meta_v2()))
    sha = hashlib.sha256(b"vk").hexdigest()
    manifest = ProtocolManifest("policy", "margin", (ManifestPath("cheap", 5.0, sha, "0x1234"),
                                                     ManifestPath("full", None, sha, "0x1234")),
                                protocol_version=2)
    mpath = tmp_path / "manifest.json"
    manifest.save(mpath)
    cert = tmp_path / "cert.json"
    v2proto.make_single_proof_certificate(str(cert), policy_digest="policy", chosen_path="cheap",
                                          attempt={"path_name": "cheap", "proof": str(proof), "vk": str(vk),
                                                   "public_inputs": str(pi), "statement_meta": str(meta)})
    return cert, mpath


@pytest.fixture
def stub_native(monkeypatch):
    monkeypatch.setattr(v2proto, "_run_native_verify", lambda cmd, attempt: (True, "stub"))


def test_v2_verifier_accepts_fresh_context(tmp_path, stub_native):
    ctx = compute_context("s1", b"n1", 1, "00")
    cert, man = _setup_certificate(tmp_path, ctx)
    seen = set()
    rep = v2proto.verify_single_proof_certificate(str(cert), str(man), "unused", expected_context=ctx,
                                                  seen_contexts=seen)
    assert rep.valid, rep.errors
    assert rep.route_margin == 20 and ctx in seen


def test_v2_verifier_rejects_other_session_and_replay(tmp_path, stub_native):
    ctx = compute_context("s1", b"n1", 1, "00")
    cert, man = _setup_certificate(tmp_path, ctx)
    other = compute_context("s2", b"n1", 1, "00")
    rep = v2proto.verify_single_proof_certificate(str(cert), str(man), "unused", expected_context=other)
    assert not rep.valid and any("context mismatch" in e for e in rep.errors)
    seen = {ctx}
    rep = v2proto.verify_single_proof_certificate(str(cert), str(man), "unused", seen_contexts=seen)
    assert not rep.valid and any("replay" in e for e in rep.errors)


def test_v2_verifier_enforces_threshold(tmp_path, stub_native):
    ctx = compute_context("s1", b"n1", 1, "00")
    cert, man = _setup_certificate(tmp_path, ctx, outputs=(12, 10, 1))   # margin 2 < 5
    rep = v2proto.verify_single_proof_certificate(str(cert), str(man), "unused")
    assert not rep.valid and any("threshold" in e for e in rep.errors)


def test_public_input_binary_roundtrip(tmp_path):
    # binary public-input layout written by zkml-prove (int32 count + 4x u64 limbs)
    from zkml.statement_v2 import load_public_inputs_any
    p = tmp_path / "pi.bin"
    vals = [1, R - 1, 123456789]
    with open(p, "wb") as f:
        f.write(struct.pack("<i", len(vals)))
        for v in vals:
            f.write(struct.pack("<4Q", *[(v >> (64 * i)) & ((1 << 64) - 1) for i in range(4)]))
    assert load_public_inputs_any(str(p)) == vals
