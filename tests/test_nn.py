#!/usr/bin/env python3
"""
Neural network quantization and inference tests.
Tests the Python-side model loading, quantization, and inference pipeline.
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from zkml.model import Model
from quantize import (
    quantize_symmetric, dequantize_symmetric,
    quantize_to_field, field_to_int, int_to_field,
    verify_quantization, BN254_P
)

tests_passed = 0
tests_failed = 0


def check_case(name, condition):
    global tests_passed, tests_failed
    status = "PASS" if condition else "FAIL"
    print(f"  {status}: {name}")
    if condition:
        tests_passed += 1
    else:
        tests_failed += 1


def test_quantization():
    print("\n=== Quantization Tests ===")
    np.random.seed(42)

    # Basic symmetric quantization
    data = np.array([1.0, -0.5, 0.25, -1.0, 0.0], dtype=np.float32)
    q, scale = quantize_symmetric(data, bits=8)
    check_case("quantize range", q.max() <= 127 and q.min() >= -127)
    check_case("zero maps to zero", q[4] == 0)

    # Round-trip
    recovered = dequantize_symmetric(q, scale)
    max_err = np.abs(data - recovered).max()
    check_case(f"round-trip error < 0.01 (got {max_err:.4f})", max_err < 0.01)

    # Field mapping
    check_case("positive int to field", int_to_field(42) == 42)
    check_case("zero to field", int_to_field(0) == 0)
    check_case("negative int to field", int_to_field(-1) == BN254_P - 1)
    check_case("field to int (positive)", field_to_int(42) == 42)
    check_case("field to int (negative)", field_to_int(BN254_P - 1) == -1)

    # Full pipeline
    data2 = np.random.randn(10).astype(np.float32)
    field_data, scale = quantize_to_field(data2, bits=8)
    check_case("field values are positive", all(v >= 0 for v in field_data.flat))
    check_case("field values < p", all(v < BN254_P for v in field_data.flat))

    passed, max_err = verify_quantization(data2, field_data, scale, tolerance=0.05)
    check_case(f"quantization verification (max_err={max_err:.4f})", passed)


def test_field_matmul():
    print("\n=== Field Matrix Multiply Tests ===")

    # Small matrix multiply
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    x = np.array([5, 6], dtype=np.float32)
    expected = A @ x  # [17, 39]

    # Quantize
    A_field, s_A = quantize_to_field(A, bits=16)
    x_field, s_x = quantize_to_field(x, bits=16)

    # Field matmul
    y_field = np.empty(2, dtype=object)
    for i in range(2):
        acc = 0
        for j in range(2):
            acc = (acc + int(A_field[i, j]) * int(x_field[j])) % BN254_P
        y_field[i] = acc

    # Recover
    y_rec = np.array([field_to_int(int(v)) for v in y_field]) / (s_A * s_x)
    err = np.abs(expected - y_rec).max()
    check_case(f"2x2 matmul error < 0.01 (got {err:.6f})", err < 0.01)

    # Larger matmul
    np.random.seed(42)
    M, N = 10, 20
    W = np.random.randn(M, N).astype(np.float32) * 0.1
    v = np.random.randn(N).astype(np.float32)
    expected = W @ v

    W_f, s_W = quantize_to_field(W, bits=16)
    v_f, s_v = quantize_to_field(v, bits=16)

    y_f = np.empty(M, dtype=object)
    for i in range(M):
        acc = 0
        for j in range(N):
            acc = (acc + int(W_f[i, j]) * int(v_f[j])) % BN254_P
        y_f[i] = acc

    y_r = np.array([field_to_int(int(v)) for v in y_f]) / (s_W * s_v)
    err = np.abs(expected - y_r).max()
    check_case(f"10x20 matmul max error: {err:.4f}", err < 0.5)


def test_model():
    print("\n=== Model Tests ===")

    # Create MNIST MLP
    model = Model.create_mnist_mlp()
    check_case("model creation", model is not None)
    check_case("input size", model.input_size == 784)
    check_case("output size", model.output_size == 10)
    check_case("num layers", len(model.layers) == 4)

    # Float inference
    np.random.seed(42)
    x = np.random.randn(784).astype(np.float32)
    y = model.float_inference(x)
    check_case("inference output shape", y.shape == (10,))
    check_case("softmax sums to ~1", abs(y.sum() - 1.0) < 0.01)

    # Model save/load
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name

    try:
        model.to_binary(path)
        check_case("model save", os.path.exists(path))

        arch = [
            ("linear", 784, 128),
            ("relu", 128, 128),
            ("linear", 128, 10),
            ("softmax", 10, 10),
        ]
        model2 = Model.from_binary(path, arch)
        check_case("model load", model2 is not None)
        check_case("loaded input size", model2.input_size == 784)
    finally:
        os.unlink(path)


def test_architecture_sidecar():
    print("\n=== Architecture Sidecar Tests ===")

    model = Model.create_cifar_mlp()
    check_case("cifar input size", model.input_size == 3072)
    check_case("cifar output size", model.output_size == 10)
    check_case("cifar layer count", len(model.layers) == 6)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".arch.json", delete=False) as f:
        arch_path = f.name

    try:
        model.save_architecture(arch_path)
        loaded = Model.from_architecture_file(arch_path)
        check_case("architecture reload", loaded is not None)
        check_case("reloaded input size", loaded.input_size == 3072)
        check_case("reloaded layer count", len(loaded.layers) == len(model.layers))
        check_case(
            "softmax remains final",
            loaded.layers[-1].layer_type == "softmax" and all(
                layer.layer_type != "softmax" for layer in loaded.layers[:-1]
            ),
        )
    finally:
        os.unlink(arch_path)


def test_tiny_transformer():
    print("\n=== Tiny Transformer Tests ===")

    model = Model.create_tiny_transformer()
    check_case("transformer input size", model.input_size == 32)
    check_case("transformer output size", model.output_size == 4)
    check_case("transformer has attention", model.layers[0].layer_type == "self_attention")

    np.random.seed(7)
    x = np.random.randn(32).astype(np.float32)
    y = model.float_inference(x)
    check_case("transformer output shape", y.shape == (4,))
    check_case("transformer softmax sums to ~1", abs(float(y.sum()) - 1.0) < 0.05)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    arch_path = path + ".arch.json"

    try:
        model.to_binary(path)
        model.save_architecture(arch_path)
        model2 = Model.from_binary(
            path,
            model.architecture_spec(),
        )
        check_case("transformer binary reload", model2 is not None)
    finally:
        os.unlink(path)
        if os.path.exists(arch_path):
            os.unlink(arch_path)


def test_multihead_transformer():
    print("\n=== Multi-Head Transformer Tests ===")

    model = Model.create_multihead_transformer(num_heads=2, num_blocks=2)
    attention_layers = [layer for layer in model.layers if layer.layer_type == "self_attention"]
    check_case("multihead transformer input size", model.input_size == 32)
    check_case("multihead transformer output size", model.output_size == 4)
    check_case("multihead transformer attention depth", len(attention_layers) == 2)
    check_case(
        "multihead transformer head count",
        all(int(layer.config.get("num_heads", 1)) == 2 for layer in attention_layers),
    )

    np.random.seed(17)
    x = np.random.randn(32).astype(np.float32)
    y = model.float_inference(x)
    check_case("multihead transformer output shape", y.shape == (4,))
    check_case("multihead transformer softmax sums to ~1", abs(float(y.sum()) - 1.0) < 0.05)

    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
        path = f.name
    arch_path = path + ".arch.json"

    try:
        model.to_binary(path)
        model.save_architecture(arch_path)
        loaded = Model.from_architecture_file(arch_path)
        check_case("multihead architecture reload", loaded is not None)
        attn_layers = [layer for layer in loaded.layers if layer.layer_type == "self_attention"]
        check_case(
            "multihead architecture head metadata",
            all(int(layer.config.get("num_heads", 1)) == 2 for layer in attn_layers),
        )
        model2 = Model.from_binary(path, model.architecture_spec())
        check_case("multihead transformer binary reload", model2 is not None)
    finally:
        os.unlink(path)
        if os.path.exists(arch_path):
            os.unlink(arch_path)


def test_accuracy():
    print("\n=== Accuracy Comparison: float32 vs quantized ===")

    model = Model.create_mnist_mlp()
    np.random.seed(42)

    # Run multiple inputs
    n_correct_float = 0
    n_correct_quant = 0
    n_agree = 0
    n_total = 100


    for i in range(n_total):
        x = np.random.randn(784).astype(np.float32)

        # Float inference
        y_float = model.float_inference(x)
        pred_float = np.argmax(y_float)

        # Quantized inference (full pipeline)
        inp = x.copy()
        for layer in model.layers:
            if layer.layer_type == "linear":
                W_q, s_W = quantize_symmetric(layer.weights, bits=8)
                inp_q, s_inp = quantize_symmetric(inp, bits=8)
                y_q = (W_q.astype(np.int64) @ inp_q.astype(np.int64))
                inp = y_q.astype(np.float32) / (s_W * s_inp)
                if layer.bias is not None:
                    inp = inp + layer.bias
            elif layer.layer_type == "relu":
                inp = np.maximum(inp, 0)
            elif layer.layer_type == "softmax":
                e = np.exp(inp - np.max(inp))
                inp = e / np.sum(e)
        pred_quant = np.argmax(inp)
        if pred_quant == pred_float:
            n_agree += 1

    agreement = n_agree / n_total * 100
    print(f"  Float vs Quantized agreement: {agreement:.1f}% ({n_agree}/{n_total})")
    check_case(f"agreement > 80%", agreement > 80)


if __name__ == "__main__":
    print("CUDA-zkML Neural Network Tests")
    print("=" * 40)

    test_quantization()
    test_field_matmul()
    test_model()
    test_architecture_sidecar()
    test_tiny_transformer()
    test_multihead_transformer()
    test_accuracy()

    print(f"\n{'=' * 40}")
    print(f"Results: {tests_passed} passed, {tests_failed} failed")
    sys.exit(1 if tests_failed > 0 else 0)
