"""
Python interface to the CUDA-zkML prover.

Wraps the C++/CUDA proving system via subprocess calls to the CLI tools,
or via ctypes/pybind11 if the shared library is available.
"""

import subprocess
import tempfile
import os
import json
import struct
import shutil
import uuid
import numpy as np
from typing import Optional, List, Tuple, Union
from contextlib import contextmanager
from .model import Model
from .artifacts import (
    limbs_to_int,
    parse_proof_bytes,
    parse_verification_key_bytes,
)


@contextmanager
def _workspace_tempdir():
    """Create a temporary directory in a writable location."""
    candidates = [
        os.environ.get("ZKML_TMPDIR"),
        os.getcwd(),
        os.path.join(os.getcwd(), ".zkml_tmp"),
        tempfile.gettempdir(),
    ]

    for base in candidates:
        if not base:
            continue
        try:
            os.makedirs(base, exist_ok=True)
            tmpdir = os.path.join(base, f"zkml_{uuid.uuid4().hex}")
            os.makedirs(tmpdir, exist_ok=False)
            try:
                yield tmpdir
            finally:
                shutil.rmtree(tmpdir, ignore_errors=True)
            return
        except (OSError, PermissionError):
            continue

    raise RuntimeError("Could not create a writable temporary directory")


class Proof:
    """Represents a Groth16 proof."""

    def __init__(self, data: bytes, public_inputs: List[Union[int, Tuple[int, int, int, int]]]):
        self.data = data
        self.public_inputs: List[Tuple[int, int, int, int]] = []
        for value in public_inputs:
            if isinstance(value, int):
                self.public_inputs.append((int(value), 0, 0, 0))
            else:
                if len(value) != 4:
                    raise ValueError("public input limbs must have length 4")
                self.public_inputs.append(
                    (int(value[0]), int(value[1]), int(value[2]), int(value[3]))
                )
        self.size_bytes = len(data)

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.data)

    @staticmethod
    def load(path: str) -> "Proof":
        with open(path, "rb") as f:
            data = f.read()
        return Proof(data, [])

    def __repr__(self):
        return f"Proof({self.size_bytes} bytes, {len(self.public_inputs)} public inputs)"

    def public_inputs_as_ints(self) -> List[int]:
        return [limbs_to_int(limbs) for limbs in self.public_inputs]

    def to_solidity_dict(self) -> dict:
        return parse_proof_bytes(self.data)


class VerificationKey:
    """Represents a Groth16 verification key."""

    def __init__(self, data: bytes):
        self.data = data

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.data)

    @staticmethod
    def load(path: str) -> "VerificationKey":
        with open(path, "rb") as f:
            return VerificationKey(f.read())

    def to_solidity_dict(self) -> dict:
        return parse_verification_key_bytes(self.data)


class Prover:
    """
    GPU-accelerated ZK prover for neural network inference.

    Usage:
        model = Model.from_onnx("model.onnx")
        prover = Prover(model)
        proof = prover.prove(input_data)
        assert prover.verify(proof)
    """

    def __init__(self, model: Model, cli_path: Optional[str] = None):
        self.model = model

        # Find CLI tools
        if cli_path:
            self.prove_cmd = os.path.join(cli_path, "zkml-prove")
            self.verify_cmd = os.path.join(cli_path, "zkml-verify")
        else:
            # Search common paths
            for path in [
                "./build",
                "../build",
                "./build/Release",
                "./cmake-build-release",
                "./cmake-build-release/Release",
                "./cmake-build-release2",
                "./cmake-build-release2/Release",
            ]:
                prove_path = os.path.join(path, "zkml-prove")
                if os.path.exists(prove_path) or os.path.exists(prove_path + ".exe"):
                    self.prove_cmd = prove_path
                    self.verify_cmd = os.path.join(path, "zkml-verify")
                    break
            else:
                self.prove_cmd = "zkml-prove"
                self.verify_cmd = "zkml-verify"

        self.vk: Optional[VerificationKey] = None

    def prove(self, input_data: np.ndarray) -> Proof:
        """
        Generate a ZK proof of correct inference.

        Args:
            input_data: Input tensor (float32), e.g., MNIST image (784,)

        Returns:
            Proof object containing the Groth16 proof
        """
        input_data = input_data.astype(np.float32).flatten()

        with _workspace_tempdir() as tmpdir:
            # Save model weights
            model_path = os.path.join(tmpdir, "model.bin")
            arch_path = model_path + ".arch.json"
            self.model.to_binary(model_path)
            self.model.save_architecture(arch_path)

            # Save input
            input_path = os.path.join(tmpdir, "input.bin")
            input_data.tofile(input_path)

            # Output paths
            proof_path = os.path.join(tmpdir, "proof.bin")
            vk_path = os.path.join(tmpdir, "vk.bin")
            pi_path = os.path.join(tmpdir, "public_inputs.bin")

            # Run prover
            cmd = [
                self.prove_cmd,
                "--model", model_path,
                "--arch", arch_path,
                "--input", input_path,
                "--output", proof_path,
                "--vk", vk_path,
                "--public-inputs", pi_path,
            ]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=300
                )
                if result.returncode != 0:
                    raise RuntimeError(
                        f"Prover failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
                    )
            except FileNotFoundError:
                raise RuntimeError(
                    f"Prover binary not found at {self.prove_cmd}. "
                    "Build the project first: mkdir build && cd build && cmake .. && make"
                )

            # Read outputs
            with open(proof_path, "rb") as f:
                proof_data = f.read()
            with open(vk_path, "rb") as f:
                self.vk = VerificationKey(f.read())

            # Read public inputs
            public_inputs = []
            with open(pi_path, "rb") as f:
                n = struct.unpack("<i", f.read(4))[0]
                for _ in range(n):
                    limbs = struct.unpack("<4Q", f.read(32))
                    public_inputs.append((limbs[0], limbs[1], limbs[2], limbs[3]))

            return Proof(proof_data, public_inputs)

    def verify(self, proof: Proof, vk: Optional[VerificationKey] = None) -> bool:
        """
        Verify a ZK proof.

        Args:
            proof: The proof to verify
            vk: Verification key (uses stored key if None)

        Returns:
            True if proof is valid
        """
        if vk is None:
            vk = self.vk
        if vk is None:
            raise ValueError("No verification key available. Run prove() first.")

        with _workspace_tempdir() as tmpdir:
            proof_path = os.path.join(tmpdir, "proof.bin")
            vk_path = os.path.join(tmpdir, "vk.bin")
            pi_path = os.path.join(tmpdir, "public_inputs.bin")

            proof.save(proof_path)
            vk.save(vk_path)

            # Write public inputs
            with open(pi_path, "wb") as f:
                f.write(struct.pack("<i", len(proof.public_inputs)))
                for limbs in proof.public_inputs:
                    f.write(struct.pack("<4Q", limbs[0], limbs[1], limbs[2], limbs[3]))

            cmd = [
                self.verify_cmd,
                "--vk", vk_path,
                "--proof", proof_path,
                "--public-inputs", pi_path,
            ]

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=60
                )
                return result.returncode == 0
            except FileNotFoundError:
                raise RuntimeError(
                    f"Verifier binary not found at {self.verify_cmd}"
                )

    def prove_and_verify(self, input_data: np.ndarray) -> dict:
        """
        Prove and verify in one step. Returns timing info.
        """
        import time

        t0 = time.time()
        proof = self.prove(input_data)
        prove_time = time.time() - t0

        t0 = time.time()
        valid = self.verify(proof)
        verify_time = time.time() - t0

        return {
            "proof": proof,
            "valid": valid,
            "prove_time_ms": prove_time * 1000,
            "verify_time_ms": verify_time * 1000,
            "proof_size_bytes": proof.size_bytes,
            "public_inputs": proof.public_inputs,
        }
