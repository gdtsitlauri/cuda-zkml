"""
CUDA-zkML: GPU-Accelerated Zero-Knowledge Proofs for Neural Network Inference

This package provides Python bindings for the CUDA-zkML prover system.
It allows you to:
1. Load PyTorch/ONNX models
2. Quantize weights to finite field elements
3. Generate ZK proofs of correct inference
4. Verify proofs

Usage:
    from zkml import Prover, Model

    model = Model.from_onnx("model.onnx")
    prover = Prover(model)
    proof = prover.prove(input_data)
    assert prover.verify(proof)
"""

__version__ = "0.1.0"
__author__ = "CUDA-zkML Team"

from .artifacts import (
    ProofArtifact,
    VerificationKeyArtifact,
    evm_curve_compatibility_report,
    export_solidity_bundle,
    load_public_inputs,
)
from .prover import Prover, Proof, VerificationKey
from .model import Model

__all__ = [
    "Model",
    "Proof",
    "ProofArtifact",
    "Prover",
    "VerificationKey",
    "VerificationKeyArtifact",
    "evm_curve_compatibility_report",
    "export_solidity_bundle",
    "load_public_inputs",
]
