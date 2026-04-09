#!/usr/bin/env python3
"""
Solidity artifact export test.

Validates that a generated proof bundle can be converted into Solidity-friendly
JSON without losing structural information.
"""

import json
import os
import shutil
import subprocess
import sys
import uuid
from contextlib import contextmanager

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from zkml import Model, export_solidity_bundle


@contextmanager
def workspace_tempdir():
    base = os.environ.get("ZKML_TMPDIR")
    if not base:
        base = os.path.join(os.path.dirname(__file__), "..", ".zkml_tmp")
    os.makedirs(base, exist_ok=True)
    tmpdir = os.path.join(base, f"zkml_solidity_{uuid.uuid4().hex}")
    os.makedirs(tmpdir, exist_ok=False)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def find_binaries():
    for path in [
        "../build",
        "../build/Release",
        "../cmake-build-release",
        "../cmake-build-release/Release",
        "../cmake-build-release2",
        "../cmake-build-release2/Release",
        ".",
    ]:
        full = os.path.join(os.path.dirname(__file__), path)
        prove = os.path.join(full, "zkml-prove")
        if os.path.exists(prove) or os.path.exists(prove + ".exe"):
            return full
    return None


def test_solidity_export_bundle():
    build_path = find_binaries()
    if not build_path:
        return

    prove_cmd = os.path.join(build_path, "zkml-prove")

    with workspace_tempdir() as tmpdir:
        model = Model.create_mnist_mlp()
        model_path = os.path.join(tmpdir, "model.bin")
        model.to_binary(model_path)

        input_data = np.random.randn(784).astype(np.float32)
        input_path = os.path.join(tmpdir, "input.bin")
        input_data.tofile(input_path)

        proof_path = os.path.join(tmpdir, "proof.bin")
        vk_path = os.path.join(tmpdir, "vk.bin")
        pi_path = os.path.join(tmpdir, "public_inputs.bin")
        export_dir = os.path.join(tmpdir, "solidity")

        result = subprocess.run(
            [
                prove_cmd,
                "--model",
                model_path,
                "--input",
                input_path,
                "--output",
                proof_path,
                "--vk",
                vk_path,
                "--public-inputs",
                pi_path,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )

        assert result.returncode == 0, result.stderr or result.stdout

        bundle = export_solidity_bundle(vk_path, proof_path, pi_path, export_dir)

        assert os.path.exists(os.path.join(export_dir, "vk.solidity.json"))
        assert os.path.exists(os.path.join(export_dir, "proof.solidity.json"))
        assert os.path.exists(os.path.join(export_dir, "public_inputs.solidity.json"))
        assert os.path.exists(os.path.join(export_dir, "calldata.solidity.json"))
        assert os.path.exists(os.path.join(export_dir, "verifier_call.txt"))

        assert len(bundle["proof"]["a"]) == 2
        assert len(bundle["proof"]["b"]) == 2
        assert len(bundle["proof"]["b"][0]) == 2
        assert len(bundle["proof"]["c"]) == 2
        assert len(bundle["vk"]["ic"]) == int(bundle["vk"]["numPublic"]) + 1
        assert len(bundle["publicInputs"]) == int(bundle["vk"]["numPublic"])
        assert all(isinstance(v, str) and v for v in bundle["publicInputs"])

        with open(os.path.join(export_dir, "calldata.solidity.json"), "r", encoding="utf-8") as f:
            calldata = json.load(f)
        assert calldata["verifyProofArgs"]["proof"]["a"] == bundle["proof"]["a"]
        assert calldata["verifyProofArgs"]["publicInputs"] == bundle["publicInputs"]
