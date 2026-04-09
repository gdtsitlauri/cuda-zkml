#!/usr/bin/env python3
"""
Float → Finite Field Quantization for zkML

Converts float32 neural network weights and activations to quantized
integers, then maps them into the BN254 finite field.

Quantization strategies:
1. Symmetric: x_q = round(x * s), where s = (2^{b-1} - 1) / max(|x|)
2. Asymmetric: x_q = round(x * s) + z, with zero-point z

For ZK proving, we use symmetric quantization to maintain
the homomorphic property: q(a + b) = q(a) + q(b) (approximately).
"""

import numpy as np
from typing import Tuple, Dict


# BN254 prime field modulus
BN254_P = 21888242871839275222246405745257275088696311157297823662689037894645226208583


def quantize_symmetric(
    data: np.ndarray,
    bits: int = 8,
    max_val: float = None
) -> Tuple[np.ndarray, float]:
    """
    Symmetric quantization: maps float to [-2^{b-1}+1, 2^{b-1}-1].

    Args:
        data: float32 array
        bits: quantization bitwidth (8 or 16)
        max_val: max absolute value for scaling (auto-detected if None)

    Returns:
        (quantized_int_array, scale_factor)
    """
    if max_val is None:
        max_val = np.abs(data).max()
    if max_val == 0:
        max_val = 1.0

    qmax = (1 << (bits - 1)) - 1
    scale = qmax / max_val

    quantized = np.round(data * scale).astype(np.int32)
    quantized = np.clip(quantized, -qmax, qmax)

    return quantized, scale


def dequantize_symmetric(
    quantized: np.ndarray,
    scale: float
) -> np.ndarray:
    """Inverse of symmetric quantization."""
    return quantized.astype(np.float32) / scale


def int_to_field(x: int) -> int:
    """
    Map an integer to BN254 field element.
    Negative integers become p - |x| (additive inverse).
    """
    if x >= 0:
        return x % BN254_P
    else:
        return (BN254_P + x) % BN254_P


def field_to_int(x: int, max_val: int = 2**31) -> int:
    """
    Map a BN254 field element back to a signed integer.
    Values close to p are interpreted as negative.
    """
    if x > BN254_P // 2:
        return x - BN254_P
    return x


def quantize_to_field(
    data: np.ndarray,
    bits: int = 8,
    max_val: float = None
) -> Tuple[np.ndarray, float]:
    """
    Full pipeline: float32 → int → BN254 field element.

    Returns:
        (field_elements as Python-int array, scale_factor)
    """
    quantized, scale = quantize_symmetric(data, bits, max_val)

    # BN254 elements can exceed int64 range, so store as Python ints.
    field_data = np.empty(quantized.shape, dtype=object)
    for idx in np.ndindex(quantized.shape):
        field_data[idx] = int_to_field(int(quantized[idx]))

    return field_data, scale


def quantize_model(
    weights: Dict[str, np.ndarray],
    bits: int = 8
) -> Dict[str, dict]:
    """
    Quantize all model weights.

    Args:
        weights: dict of layer_name -> weight_array
        bits: quantization bitwidth

    Returns:
        dict of layer_name -> {"field_data": array, "scale": float, "shape": tuple}
    """
    result = {}
    for name, data in weights.items():
        field_data, scale = quantize_to_field(data, bits)
        result[name] = {
            "field_data": field_data,
            "scale": scale,
            "shape": data.shape,
            "bits": bits,
        }
        print(f"  {name}: shape={data.shape}, scale={scale:.4f}, "
              f"range=[{data.min():.4f}, {data.max():.4f}]")
    return result


def verify_quantization(
    original: np.ndarray,
    field_data: np.ndarray,
    scale: float,
    tolerance: float = 0.1
) -> Tuple[bool, float]:
    """
    Verify quantization accuracy.

    Returns:
        (passed, max_error)
    """
    # Convert back
    recovered = np.zeros(original.shape, dtype=np.float32)
    for idx in np.ndindex(field_data.shape):
        int_val = field_to_int(int(field_data[idx]))
        recovered[idx] = int_val / scale

    max_error = np.abs(original - recovered).max()
    mean_error = np.abs(original - recovered).mean()

    passed = max_error < tolerance
    print(f"  Quantization error: max={max_error:.6f}, mean={mean_error:.6f}, "
          f"{'PASS' if passed else 'FAIL'}")

    return passed, max_error


if __name__ == "__main__":
    print("Quantization Test")
    print("=================")

    # Test with random data
    np.random.seed(42)
    data = np.random.randn(10, 5).astype(np.float32) * 0.5

    print("\nOriginal data range:", data.min(), "to", data.max())

    # Quantize
    field_data, scale = quantize_to_field(data, bits=8)
    print(f"Scale factor: {scale}")
    print(f"Field data sample: {field_data[0, :3]}")

    # Verify round-trip
    passed, max_err = verify_quantization(data, field_data, scale, tolerance=0.01)

    # Test matrix multiply in field
    print("\nField arithmetic test:")
    A = np.array([[1, 2], [3, 4]], dtype=np.float32)
    x = np.array([5, 6], dtype=np.float32)

    # Float result
    y_float = A @ x
    print(f"Float result: {y_float}")

    # Field result
    A_field, s_A = quantize_to_field(A, bits=16)
    x_field, s_x = quantize_to_field(x, bits=16)

    # Matrix multiply in field (mod p)
    y_field = np.empty(2, dtype=object)
    for i in range(2):
        acc = 0
        for j in range(2):
            acc = (acc + int(A_field[i, j]) * int(x_field[j])) % BN254_P
        y_field[i] = acc

    # Convert back
    y_recovered = np.array([field_to_int(int(v)) for v in y_field]) / (s_A * s_x)
    print(f"Field result (recovered): {y_recovered}")
    print(f"Error: {np.abs(y_float - y_recovered).max():.6f}")
