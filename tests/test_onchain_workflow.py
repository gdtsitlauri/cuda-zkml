#!/usr/bin/env python3

import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from contextlib import contextmanager

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from zkml.artifacts import (
    ProofArtifact,
    VerificationKeyArtifact,
    evm_curve_compatibility_report,
)


@contextmanager
def workspace_tempdir():
    base = os.environ.get("ZKML_TMPDIR")
    if not base:
        base = os.path.join(os.path.dirname(__file__), "..", ".zkml_tmp")
    os.makedirs(base, exist_ok=True)
    tmpdir = os.path.join(base, f"zkml_onchain_{uuid.uuid4().hex}")
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


def find_solc_binary():
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", ".solcx", "solc-v0.8.20", "solc.exe"),
        os.path.join(os.path.dirname(__file__), "..", ".solcx", "solc-v0.8.20", "solc"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
    return None


def generate_demo_artifacts(tmpdir: str):
    build_path = find_binaries()
    if not build_path:
        pytest.skip("Build directory not found")

    prove_cmd = os.path.join(build_path, "zkml-prove")
    result = subprocess.run(
        [prove_cmd, "--demo"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd=tmpdir,
    )
    if result.returncode != 0:
        pytest.fail(result.stderr[:500] if result.stderr else result.stdout[:500])

    proof_path = os.path.join(tmpdir, "proof.bin")
    vk_path = os.path.join(tmpdir, "vk.bin")
    public_inputs_path = os.path.join(tmpdir, "public_inputs.bin")
    return proof_path, vk_path, public_inputs_path


def test_curve_report_marks_demo_artifacts_as_evm_compatible():
    with workspace_tempdir() as tmpdir:
        proof_path, vk_path, _ = generate_demo_artifacts(tmpdir)
        proof = ProofArtifact.from_file(proof_path)
        vk = VerificationKeyArtifact.from_file(vk_path)
        report = evm_curve_compatibility_report(vk, proof)

        assert report["proof"]["A"] is True
        assert report["proof"]["C"] is True
        assert report["vk"]["alpha"] is True
        assert report["proof"]["B"] is True
        assert report["vk"]["beta"] is True
        assert report["vk"]["gamma"] is True
        assert report["vk"]["delta"] is True
        assert report["all_points_evm_compatible"] is True


def test_onchain_verify_script_reports_success_cleanly():
    pytest.importorskip("web3")
    pytest.importorskip("solcx")
    pytest.importorskip("eth_tester")

    solc_binary = find_solc_binary()
    if not solc_binary:
        pytest.skip("solc binary not found under .solcx")

    with workspace_tempdir() as tmpdir:
        proof_path, vk_path, public_inputs_path = generate_demo_artifacts(tmpdir)
        report_path = os.path.join(tmpdir, "onchain_report.json")
        artifacts_dir = os.path.join(tmpdir, "solidity_export")

        script_path = os.path.join(os.path.dirname(__file__), "..", "python", "onchain_verify.py")
        result = subprocess.run(
            [
                sys.executable,
                script_path,
                "--vk",
                vk_path,
                "--proof",
                proof_path,
                "--public-inputs",
                public_inputs_path,
                "--solc-binary",
                solc_binary,
                "--output-json",
                report_path,
                "--output-dir",
                artifacts_dir,
            ],
            capture_output=True,
            text=True,
            timeout=300,
        )
        assert result.returncode == 0, result.stderr
        assert os.path.exists(report_path)
        assert os.path.exists(os.path.join(artifacts_dir, "calldata.solidity.json"))

        with open(report_path, "r", encoding="utf-8") as f:
            report = json.load(f)

        assert report["curve_compatibility"]["all_points_evm_compatible"] is True
        assert report["status"] == "verified"
        assert report["evm_valid"] is True
