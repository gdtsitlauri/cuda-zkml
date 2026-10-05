#!/usr/bin/env python3
"""Train and export a real-ReLU model for the PCANI v2 statement (roadmap 2.1).

Replaces the untrained h = z^2 + 64z + 64 activation of the v1 experiments with
a *trained* ReLU network proven exactly in-circuit (RELU_EXACT: bit-decomposed
sign + range-checked rescaling). Dataset: scikit-learn digits (8x8, 0..16).

Integer model (all values exact int32; the R1CS enforces precisely this):
  z1 = W1q x + b1q                 W1q = round(W1 * 2^F), b1q = round(b1 * 2^F)
  h  = floor(relu(z1) / 2^F)       RELU_EXACT(bits, shift=F)
  s  = W2q h + b2q                 W2q = round(W2 * 2^F), b2q = round(b2 * 2^F)
Outputs: <out>/model.int32.bin, model.arch.json, inputs/sample_<i>.npy,
expected_scores.json (exact integer scores per sample), report.json (float vs
integer accuracy). Used by the Colab notebook: zkml-prove --statement-v2
--integer-model --integer-input --arch model.arch.json.

  python python/export_trained_relu_model.py --out artifacts/relu_digits [--hidden 32] [--frac-bits 6]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from zkml.statement_v2 import reference_scores  # noqa: E402


def main() -> int:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--frac-bits", type=int, default=6)
    ap.add_argument("--relu-bits", type=int, default=16)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    X, y = load_digits(return_X_y=True)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=a.seed, stratify=y)
    clf = MLPClassifier(hidden_layer_sizes=(a.hidden,), activation="relu", max_iter=600,
                        random_state=a.seed, alpha=1e-3)
    clf.fit(Xtr, ytr)
    float_acc = float(clf.score(Xte, yte))

    F = a.frac_bits
    W1, W2 = clf.coefs_[0].T, clf.coefs_[1].T          # [hidden, 64], [10, hidden]
    b1, b2 = clf.intercepts_
    q = lambda arr: np.round(arr * (1 << F)).astype(np.int64)
    W1q, b1q, W2q, b2q = q(W1), q(b1), q(W2), q(b2)
    params = np.concatenate([W1q.ravel(), b1q, W2q.ravel(), b2q]).astype(np.int64)
    assert np.all(np.abs(params) < 2 ** 31)

    layers = [
        {"type": "linear", "in_features": 64, "out_features": a.hidden},
        {"type": "relu_exact", "in_features": a.hidden, "out_features": a.hidden,
         "bits": a.relu_bits, "shift": F},
        {"type": "linear", "in_features": a.hidden, "out_features": 10},
        {"type": "softmax", "in_features": 10, "out_features": 10},
    ]
    plist = [int(v) for v in params]
    Xi = Xte.astype(np.int64)
    int_pred = []
    max_abs_pre = 0
    for x in Xi:
        z1 = W1q @ x + b1q
        max_abs_pre = max(max_abs_pre, int(np.max(np.abs(z1))))
        s = reference_scores(layers, plist, x.tolist())
        int_pred.append(int(np.argmax(s)))
    int_acc = float(np.mean(np.array(int_pred) == yte))
    if max_abs_pre >= 2 ** (a.relu_bits - 1):
        raise SystemExit(f"pre-activations reach {max_abs_pre}; increase --relu-bits")

    os.makedirs(os.path.join(a.out, "inputs"), exist_ok=True)
    params.astype(np.int32).tofile(os.path.join(a.out, "model.int32.bin"))
    with open(os.path.join(a.out, "model.arch.json"), "w") as f:
        json.dump({"name": "digits_relu_exact", "layers": layers}, f, indent=2)
    expected = {}
    for i in range(a.samples):
        path = os.path.join(a.out, "inputs", f"sample_{i}.npy")
        np.save(path, Xi[i].astype(np.float32))
        expected[f"sample_{i}"] = {"label": int(yte[i]),
                                   "scores": reference_scores(layers, plist, Xi[i].tolist())}
    with open(os.path.join(a.out, "expected_scores.json"), "w") as f:
        json.dump(expected, f, indent=2)
    # plain-text copy for the C++ host cross-check (tests/host/test_statement_v2.cpp):
    # line 1: hidden frac_bits relu_bits n; then per sample: 64 inputs followed by 10 scores
    with open(os.path.join(a.out, "expected_scores.txt"), "w") as f:
        f.write(f"{a.hidden} {F} {a.relu_bits} {a.samples}\n")
        for i in range(a.samples):
            f.write(" ".join(str(int(v)) for v in Xi[i]) + " " +
                    " ".join(str(v) for v in expected[f"sample_{i}"]["scores"]) + "\n")
    report = {"dataset": "sklearn digits 8x8 (test split 25%)", "hidden": a.hidden, "frac_bits": F,
              "relu_bits": a.relu_bits, "float_accuracy": float_acc, "integer_accuracy": int_acc,
              "max_abs_preactivation": max_abs_pre, "params": int(params.size)}
    with open(os.path.join(a.out, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
