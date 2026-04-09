#!/usr/bin/env python3
"""
End-to-end test: Load model → Quantize → Prove → Verify → "PROOF VALID"

This test exercises the complete pipeline by calling the CLI tools.
"""

import sys
import os
import subprocess
import tempfile
import shutil
import uuid
import numpy as np
from contextlib import contextmanager

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from zkml.model import Model


@contextmanager
def workspace_tempdir():
    base = os.environ.get("ZKML_TMPDIR")
    if not base:
        base = os.path.join(os.path.dirname(__file__), "..", ".zkml_tmp")
    os.makedirs(base, exist_ok=True)
    tmpdir = os.path.join(base, f"zkml_test_{uuid.uuid4().hex}")
    os.makedirs(tmpdir, exist_ok=False)
    try:
        yield tmpdir
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def find_binaries():
    """Find compiled CLI binaries."""
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


def run_demo_mode():
    """Test the --demo mode of zkml-prove."""
    print("\n=== Test: Demo Mode ===")

    build_path = find_binaries()
    if not build_path:
        print("  SKIP: Build directory not found. Run cmake build first.")
        return True

    prove_cmd = os.path.join(build_path, "zkml-prove")

    print(f"  Running: {prove_cmd} --demo")
    with workspace_tempdir() as tmpdir:
        result = subprocess.run(
            [prove_cmd, "--demo"],
            capture_output=True, text=True, timeout=300,
            cwd=tmpdir
        )

    print(f"  Return code: {result.returncode}")
    if result.stdout:
        # Print last 20 lines
        lines = result.stdout.strip().split("\n")
        for line in lines[-20:]:
            print(f"  > {line}")

    if result.returncode != 0:
        print(f"  FAIL: Prover returned non-zero")
        if result.stderr:
            print(f"  stderr: {result.stderr[:500]}")
        return False

    # Check for expected output
    if "PROOF GENERATION COMPLETE" in result.stdout:
        print("  PASS: Proof generation completed")
    else:
        print("  FAIL: Missing completion message")
        return False

    if "Proof valid:        YES" in result.stdout:
        print("  PASS: Demo proof verified")
    else:
        print("  FAIL: Demo proof did not verify")
        return False

    return True


def run_prove_verify_pipeline():
    """Test the full prove + verify pipeline."""
    print("\n=== Test: Prove + Verify Pipeline ===")

    build_path = find_binaries()
    if not build_path:
        print("  SKIP: Build directory not found")
        return True

    prove_cmd = os.path.join(build_path, "zkml-prove")
    verify_cmd = os.path.join(build_path, "zkml-verify")

    with workspace_tempdir() as tmpdir:
        # Create model and input
        model = Model.create_mnist_mlp()
        model_path = os.path.join(tmpdir, "model.bin")
        arch_path = model_path + ".arch.json"
        model.to_binary(model_path)
        model.save_architecture(arch_path)

        input_data = np.random.randn(784).astype(np.float32)
        input_path = os.path.join(tmpdir, "input.bin")
        input_data.tofile(input_path)

        proof_path = os.path.join(tmpdir, "proof.bin")
        vk_path = os.path.join(tmpdir, "vk.bin")
        pi_path = os.path.join(tmpdir, "public_inputs.bin")

        # Prove
        print("  Step 1: Generating proof...")
        result = subprocess.run(
            [prove_cmd,
             "--model", model_path,
             "--arch", arch_path,
             "--input", input_path,
             "--output", proof_path,
             "--vk", vk_path,
             "--public-inputs", pi_path],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            print(f"  FAIL: Prover failed")
            print(result.stderr[:500] if result.stderr else "")
            return False

        # Check output files exist
        for f, name in [(proof_path, "proof"), (vk_path, "vk"), (pi_path, "public_inputs")]:
            if not os.path.exists(f):
                print(f"  FAIL: {name} file not created")
                return False
            size = os.path.getsize(f)
            print(f"  {name}: {size} bytes")

        # Verify
        print("  Step 2: Verifying proof...")
        result = subprocess.run(
            [verify_cmd,
             "--vk", vk_path,
             "--proof", proof_path,
             "--public-inputs", pi_path],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0:
            print("  PASS: PROOF VALID")
            return True
        else:
            print("  FAIL: Verification failed")
            print(result.stdout[-200:] if result.stdout else "")
            return False


def run_python_prover():
    """Test the Python prover wrapper."""
    print("\n=== Test: Python Prover Wrapper ===")

    build_path = find_binaries()
    if not build_path:
        print("  SKIP: Build directory not found")
        return True

    from zkml import Prover, Model

    model = Model.create_mnist_mlp()
    prover = Prover(model, cli_path=build_path)

    input_data = np.random.randn(784).astype(np.float32)

    try:
        result = prover.prove_and_verify(input_data)
        print(f"  Prove time: {result['prove_time_ms']:.1f} ms")
        print(f"  Verify time: {result['verify_time_ms']:.1f} ms")
        print(f"  Proof size: {result['proof_size_bytes']} bytes")
        print(f"  Valid: {result['valid']}")

        if result['valid']:
            print("  PASS: End-to-end pipeline works")
            return True
        else:
            print("  FAIL: Proof invalid")
            return False
    except Exception as e:
        print(f"  FAIL: {e}")
        return False


def run_pk_reuse_pipeline():
    """Test proving-key save/load reuse through the CLI."""
    print("\n=== Test: Proving Key Reuse ===")

    build_path = find_binaries()
    if not build_path:
        print("  SKIP: Build directory not found")
        return True

    prove_cmd = os.path.join(build_path, "zkml-prove")
    verify_cmd = os.path.join(build_path, "zkml-verify")

    with workspace_tempdir() as tmpdir:
        model = Model.create_mnist_mlp()
        model_path = os.path.join(tmpdir, "model.bin")
        arch_path = model_path + ".arch.json"
        model.to_binary(model_path)
        model.save_architecture(arch_path)

        input_a = np.random.randn(784).astype(np.float32)
        input_b = np.random.randn(784).astype(np.float32)
        input_a_path = os.path.join(tmpdir, "input_a.bin")
        input_b_path = os.path.join(tmpdir, "input_b.bin")
        input_a.tofile(input_a_path)
        input_b.tofile(input_b_path)

        pk_path = os.path.join(tmpdir, "pk.bin")
        vk_path = os.path.join(tmpdir, "vk.bin")
        pi_a_path = os.path.join(tmpdir, "public_inputs_a.bin")
        pi_b_path = os.path.join(tmpdir, "public_inputs_b.bin")
        proof_a_path = os.path.join(tmpdir, "proof_a.bin")
        proof_b_path = os.path.join(tmpdir, "proof_b.bin")

        print("  Step 1: setup + save PK")
        result = subprocess.run(
            [
                prove_cmd,
                "--model", model_path,
                "--arch", arch_path,
                "--input", input_a_path,
                "--output", proof_a_path,
                "--vk", vk_path,
                "--public-inputs", pi_a_path,
                "--pk-save", pk_path,
            ],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print("  FAIL: Initial prove with --pk-save failed")
            print(result.stderr[:500] if result.stderr else result.stdout[:500])
            return False

        if not os.path.exists(pk_path):
            print("  FAIL: PK file was not created")
            return False

        print("  Step 2: reuse PK with new input")
        result = subprocess.run(
            [
                prove_cmd,
                "--model", model_path,
                "--arch", arch_path,
                "--input", input_b_path,
                "--output", proof_b_path,
                "--vk", vk_path,
                "--public-inputs", pi_b_path,
                "--pk-load", pk_path,
            ],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print("  FAIL: Prove with --pk-load failed")
            print(result.stderr[:500] if result.stderr else result.stdout[:500])
            return False

        print("  Step 3: verify reused-PK proof")
        result = subprocess.run(
            [
                verify_cmd,
                "--vk", vk_path,
                "--proof", proof_b_path,
                "--public-inputs", pi_b_path,
            ],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode != 0:
            print("  FAIL: Verification after PK reuse failed")
            print(result.stdout[-200:] if result.stdout else "")
            return False

        print("  PASS: PK reuse pipeline works")
        return True


def run_tiny_transformer_pipeline():
    """Test the tiny transformer path through CLI prove/verify."""
    print("\n=== Test: Tiny Transformer Pipeline ===")

    build_path = find_binaries()
    if not build_path:
        print("  SKIP: Build directory not found")
        return True

    prove_cmd = os.path.join(build_path, "zkml-prove")
    verify_cmd = os.path.join(build_path, "zkml-verify")

    with workspace_tempdir() as tmpdir:
        model = Model.create_tiny_transformer()
        model_path = os.path.join(tmpdir, "transformer.bin")
        arch_path = model_path + ".arch.json"
        model.to_binary(model_path)
        model.save_architecture(arch_path)

        input_data = np.random.randn(model.input_size).astype(np.float32)
        input_path = os.path.join(tmpdir, "input.bin")
        input_data.tofile(input_path)

        proof_path = os.path.join(tmpdir, "proof.bin")
        vk_path = os.path.join(tmpdir, "vk.bin")
        pi_path = os.path.join(tmpdir, "public_inputs.bin")

        result = subprocess.run(
            [
                prove_cmd,
                "--model", model_path,
                "--arch", arch_path,
                "--input", input_path,
                "--output", proof_path,
                "--vk", vk_path,
                "--public-inputs", pi_path,
            ],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print("  FAIL: Transformer prove failed")
            print(result.stderr[:500] if result.stderr else result.stdout[:500])
            return False

        result = subprocess.run(
            [
                verify_cmd,
                "--vk", vk_path,
                "--proof", proof_path,
                "--public-inputs", pi_path,
            ],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print("  FAIL: Transformer verify failed")
            print(result.stderr[:500] if result.stderr else result.stdout[-300:])
            return False

        print("  PASS: Tiny transformer prove/verify works")
        return True


def run_multihead_transformer_pipeline():
    """Test a stacked multi-head transformer path through CLI prove/verify."""
    print("\n=== Test: Multi-Head Transformer Pipeline ===")

    build_path = find_binaries()
    if not build_path:
        print("  SKIP: Build directory not found")
        return True

    prove_cmd = os.path.join(build_path, "zkml-prove")
    verify_cmd = os.path.join(build_path, "zkml-verify")

    with workspace_tempdir() as tmpdir:
        model = Model.create_multihead_transformer(num_heads=2, num_blocks=2)
        model_path = os.path.join(tmpdir, "transformer_multihead.bin")
        arch_path = model_path + ".arch.json"
        model.to_binary(model_path)
        model.save_architecture(arch_path)

        input_data = np.random.randn(model.input_size).astype(np.float32)
        input_path = os.path.join(tmpdir, "input.bin")
        input_data.tofile(input_path)

        proof_path = os.path.join(tmpdir, "proof.bin")
        vk_path = os.path.join(tmpdir, "vk.bin")
        pi_path = os.path.join(tmpdir, "public_inputs.bin")

        result = subprocess.run(
            [
                prove_cmd,
                "--model", model_path,
                "--arch", arch_path,
                "--input", input_path,
                "--output", proof_path,
                "--vk", vk_path,
                "--public-inputs", pi_path,
            ],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode != 0:
            print("  FAIL: Multi-head transformer prove failed")
            print(result.stderr[:500] if result.stderr else result.stdout[:500])
            return False

        result = subprocess.run(
            [
                verify_cmd,
                "--vk", vk_path,
                "--proof", proof_path,
                "--public-inputs", pi_path,
            ],
            capture_output=True, text=True, timeout=120
        )
        if result.returncode != 0:
            print("  FAIL: Multi-head transformer verify failed")
            print(result.stderr[:500] if result.stderr else result.stdout[-300:])
            return False

        print("  PASS: Multi-head transformer prove/verify works")
        return True


def main():
    print("CUDA-zkML End-to-End Test Suite")
    print("=" * 50)

    results = []
    results.append(("Demo mode", run_demo_mode()))
    results.append(("Prove+Verify", run_prove_verify_pipeline()))
    results.append(("Python wrapper", run_python_prover()))
    results.append(("PK reuse", run_pk_reuse_pipeline()))
    results.append(("Tiny transformer", run_tiny_transformer_pipeline()))
    results.append(("Multi-head transformer", run_multihead_transformer_pipeline()))

    print("\n" + "=" * 50)
    all_passed = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {status}: {name}")
        if not passed:
            all_passed = False

    print(f"\n{'All tests passed!' if all_passed else 'Some tests failed.'}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())


def test_demo_mode():
    assert run_demo_mode()


def test_prove_verify_pipeline():
    assert run_prove_verify_pipeline()


def test_python_prover():
    assert run_python_prover()


def test_pk_reuse_pipeline():
    assert run_pk_reuse_pipeline()


def test_tiny_transformer_pipeline():
    assert run_tiny_transformer_pipeline()


def test_multihead_transformer_pipeline():
    assert run_multihead_transformer_pipeline()
