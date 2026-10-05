#!/usr/bin/env python3
"""Prepare an end-to-end exact-integer PCANI benchmark on sklearn Digits.

The exported candidate models use the same arithmetic as CUDA-zkML's
--integer-model/--integer-input mode:

    z = W x + b
    h = z^2 + 64 z + 64
    logits = V h + c

All model parameters are int32 and all benchmark inputs are integer-valued.
Therefore the Python logits saved here can be compared exactly with signed
public outputs from the fixed-model Groth16 statement.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def hidden_features(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = X.astype(np.int64) @ W.astype(np.int64).T + b.astype(np.int64)
    return z * z + 64 * z + 64


def train_output_layer(H_train: np.ndarray, y_train: np.ndarray):
    scaler = StandardScaler()
    Hs = scaler.fit_transform(H_train.astype(np.float64))
    clf = LogisticRegression(max_iter=5000, C=10.0, random_state=2026)
    clf.fit(Hs, y_train)

    coef_raw = clf.coef_ / scaler.scale_[None, :]
    bias_raw = clf.intercept_ - (
        clf.coef_ * scaler.mean_[None, :] / scaler.scale_[None, :]
    ).sum(axis=1)

    max_c = float(np.max(np.abs(coef_raw))) or 1.0
    max_b = float(np.max(np.abs(bias_raw))) or 1.0
    # Keep a generous int32 margin while retaining all useful coefficients.
    scale = min(100_000.0 / max_c, 1_000_000_000.0 / max_b)
    W_out = np.rint(coef_raw * scale).astype(np.int64)
    b_out = np.rint(bias_raw * scale).astype(np.int64)
    if np.max(np.abs(W_out)) > np.iinfo(np.int32).max or np.max(np.abs(b_out)) > np.iinfo(np.int32).max:
        raise RuntimeError("trained exact-integer output layer exceeds int32")
    return W_out.astype(np.int32), b_out.astype(np.int32), float(scale)


def logits_exact(X, W_hidden, b_hidden, W_out, b_out):
    H = hidden_features(X, W_hidden, b_hidden)
    # Python int64 is sufficient for this benchmark; assert before returning.
    scores = H.astype(np.int64) @ W_out.astype(np.int64).T + b_out.astype(np.int64)
    if np.max(np.abs(scores)) >= 2**62:
        raise RuntimeError("score magnitude too large for safe int64 benchmark arithmetic")
    return scores.astype(np.int64)


def write_arch(path: Path, width: int) -> None:
    payload = {
        "layers": [
            {"type": "linear", "in_features": 64, "out_features": width},
            {"type": "relu", "in_features": width, "out_features": width},
            {"type": "linear", "in_features": width, "out_features": 10},
            {"type": "softmax", "in_features": 10, "out_features": 10},
        ]
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="benchmarks/pcani_digits")
    ap.add_argument("--widths", default="16,48,96,160")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    if not widths or sorted(widths) != widths:
        raise ValueError("widths must be a non-empty increasing list")

    X, y = load_digits(return_X_y=True)
    X = np.rint(X).astype(np.int32)
    y = y.astype(np.int64)

    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.40, random_state=2026, stratify=y
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=2027, stratify=y_tmp
    )

    rng = np.random.default_rng(2026)
    max_width = max(widths)
    W_master = rng.choice(
        np.array([-2, -1, 0, 1, 2], dtype=np.int32),
        size=(max_width, 64),
        p=[0.10, 0.20, 0.40, 0.20, 0.10],
    ).astype(np.int32)
    b_master = rng.integers(-4, 5, size=max_width, dtype=np.int32)

    cal_payload = {"labels": y_cal}
    test_payload = {"labels": y_test}
    report = {
        "benchmark": "sklearn-digits exact-integer PCANI",
        "dataset": {"train": len(X_train), "calibration": len(X_cal), "test": len(X_test)},
        "arithmetic": "int32 parameters / integer input / BN254 field embedding; no float quantization",
        "activation": "h=z^2+64z+64",
        "seed": 2026,
        "paths": [],
    }

    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    for width in widths:
        name = f"p{width}"
        W_h = W_master[:width].copy()
        b_h = b_master[:width].copy()
        H_train = hidden_features(X_train, W_h, b_h)
        W_o, b_o, fit_scale = train_output_layer(H_train, y_train)
        cal_logits = logits_exact(X_cal, W_h, b_h, W_o, b_o)
        test_logits = logits_exact(X_test, W_h, b_h, W_o, b_o)
        cal_acc = float((cal_logits.argmax(axis=1) == y_cal).mean())
        test_acc = float((test_logits.argmax(axis=1) == y_test).mean())

        path_dir = models_dir / name
        path_dir.mkdir(exist_ok=True)
        model_path = path_dir / "model.i32.bin"
        arch_path = path_dir / "model.i32.bin.arch.json"
        model_values = np.concatenate([
            W_h.reshape(-1), b_h.reshape(-1), W_o.reshape(-1), b_o.reshape(-1)
        ]).astype("<i4")
        model_path.write_bytes(model_values.tobytes())
        write_arch(arch_path, width)

        cal_payload[f"{name}_logits"] = cal_logits.astype(np.float64)
        test_payload[f"{name}_logits"] = test_logits.astype(np.float64)
        report["paths"].append({
            "name": name,
            "width": width,
            "calibration_accuracy": cal_acc,
            "test_accuracy": test_acc,
            "model_sha256": sha256_file(model_path),
            "fit_integer_scale": fit_scale,
            "parameter_count": int(model_values.size),
        })

    np.savez_compressed(out / "calibration.npz", **cal_payload)
    np.savez_compressed(out / "test.npz", **test_payload)
    np.save(out / "X_test.npy", X_test.astype(np.float32))
    np.save(out / "y_test.npy", y_test)
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
