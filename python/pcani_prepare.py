#!/usr/bin/env python3
"""Prepare an exact-integer PCANI benchmark for any supported dataset/family.

Generalises ``pcani_digits_prepare.py`` (kept unchanged so the completed T4
Digits result stays reproducible) to the Gate-F "larger benchmark/model family"
requirement.  Every exported candidate path uses exactly the arithmetic of
CUDA-zkML's ``--integer-model --integer-input`` mode:

    z = W x + b
    h = z^2 + 64 z + 64
    logits = V h + c

so the saved Python logits can be compared bit-exactly with the signed public
outputs of the fixed-model Groth16 statement, as in the Digits run.

Families
--------
``mlp``   W is a fixed random sparse {-2..2} matrix (random polynomial
          features); ``--widths`` are hidden widths.  This is the Digits model
          family applied to a larger input.
``conv``  W is a fixed random {-2..2} convolution (``--kernel``/``--stride``,
          valid padding) lowered to its exact dense Toeplitz matrix, so the
          existing LINEAR circuit proves it without new CUDA kernels.
          ``--widths`` are filter counts; the hidden width is
          filters * out_h * out_w.

Only V and c are trained (logistic regression on calibration-free training
data).  Hidden weights for path k are a prefix of the master weights, so paths
are nested exactly as in the Digits benchmark.

Datasets: ``digits`` (sklearn, offline), ``mnist`` and ``fashion_mnist``
(OpenML download; train/calibration come from the official training set and
test from the official test set).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

INT32_MAX = int(np.iinfo(np.int32).max)
# Logits are stored as float64 in the .npz files (as in the Digits benchmark)
# and later compared with exact integers, so they must stay below 2**53.
FLOAT64_EXACT_LIMIT = 2**52
DATA_SEED = 2026


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


# --------------------------------------------------------------------------
# Datasets
# --------------------------------------------------------------------------

def _stratified_subsample(X, y, n, seed):
    if n is None or n <= 0 or n >= len(y):
        return X, y
    X_s, _, y_s, _ = train_test_split(X, y, train_size=n, random_state=seed, stratify=y)
    return X_s, y_s


def _avg_pool(images: np.ndarray, side: int, pool: int) -> Tuple[np.ndarray, int]:
    if pool == 1:
        return images, side
    if side % pool != 0:
        raise ValueError(f"pool={pool} must divide image side {side}")
    new = side // pool
    imgs = images.reshape(-1, new, pool, new, pool).mean(axis=(2, 4))
    return imgs.reshape(len(images), new * new), new


def _quantize_pixels(images: np.ndarray, src_max: float, levels: int) -> np.ndarray:
    return np.rint(images.astype(np.float64) * (levels / src_max)).astype(np.int32)


def load_dataset(name: str, *, pool: int, levels: int, max_train: int, max_cal: int,
                 max_test: int, data_home: str | None):
    """Return integer splits plus image side.  Inputs are integers in [0, levels]."""
    if name == "digits":
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
        side = 8
        X, side = _avg_pool(X, side, pool)
        X = _quantize_pixels(X, 16.0, levels)
        y = y.astype(np.int64)
        # Same split procedure as pcani_digits_prepare.py.
        X_train, X_tmp, y_train, y_tmp = train_test_split(
            X, y, test_size=0.40, random_state=2026, stratify=y)
        X_cal, X_test, y_cal, y_test = train_test_split(
            X_tmp, y_tmp, test_size=0.50, random_state=2027, stratify=y_tmp)
        source = "sklearn.datasets.load_digits"
    elif name in {"mnist", "fashion_mnist"}:
        from sklearn.datasets import fetch_openml
        openml_name = {"mnist": "mnist_784", "fashion_mnist": "Fashion-MNIST"}[name]
        X, y = fetch_openml(openml_name, version=1, return_X_y=True, as_frame=False,
                            parser="liac-arff", data_home=data_home)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y).astype(np.int64)
        side = 28
        X, side = _avg_pool(X, side, pool)
        X = _quantize_pixels(X, 255.0, levels)
        # OpenML keeps the canonical order: first 60000 train, last 10000 test.
        X_official_train, y_official_train = X[:60000], y[:60000]
        X_test, y_test = X[60000:], y[60000:]
        X_train, X_cal, y_train, y_cal = train_test_split(
            X_official_train, y_official_train, test_size=10000,
            random_state=2027, stratify=y_official_train)
        source = f"OpenML {openml_name} (official 60k/10k split)"
    else:
        raise ValueError(f"unknown dataset {name}")

    X_train, y_train = _stratified_subsample(X_train, y_train, max_train, DATA_SEED)
    X_cal, y_cal = _stratified_subsample(X_cal, y_cal, max_cal, DATA_SEED + 1)
    X_test, y_test = _stratified_subsample(X_test, y_test, max_test, DATA_SEED + 2)
    return (X_train, y_train, X_cal, y_cal, X_test, y_test), side, source


# --------------------------------------------------------------------------
# Hidden-layer families
# --------------------------------------------------------------------------

def _sparse_int_weights(rng, shape):
    return rng.choice(np.array([-2, -1, 0, 1, 2], dtype=np.int32), size=shape,
                      p=[0.10, 0.20, 0.40, 0.20, 0.10]).astype(np.int32)


def conv_output_side(side: int, kernel: int, stride: int) -> int:
    if kernel > side:
        raise ValueError("kernel larger than image")
    return (side - kernel) // stride + 1


def lower_conv_to_dense(filters: np.ndarray, side: int, stride: int) -> np.ndarray:
    """Exact dense matrix of a valid-padding single-channel 2D convolution.

    Row ``f*out*out + i*out + j`` holds filter ``f`` placed at output position
    (i, j); input pixels are flattened row-major (as in sklearn/OpenML).
    """
    n_filters, k, _ = filters.shape
    out = conv_output_side(side, k, stride)
    dense = np.zeros((n_filters * out * out, side * side), dtype=np.int32)
    for f in range(n_filters):
        for i in range(out):
            for j in range(out):
                row = f * out * out + i * out + j
                for di in range(k):
                    r = (i * stride + di) * side + j * stride
                    dense[row, r:r + k] = filters[f, di]
    return dense


def direct_conv(X: np.ndarray, filters: np.ndarray, side: int, stride: int) -> np.ndarray:
    """Reference convolution used only by tests to check the lowering."""
    n_filters, k, _ = filters.shape
    out = conv_output_side(side, k, stride)
    imgs = X.reshape(len(X), side, side).astype(np.int64)
    res = np.zeros((len(X), n_filters, out, out), dtype=np.int64)
    for f in range(n_filters):
        for i in range(out):
            for j in range(out):
                patch = imgs[:, i * stride:i * stride + k, j * stride:j * stride + k]
                res[:, f, i, j] = (patch * filters[f].astype(np.int64)).sum(axis=(1, 2))
    return res.reshape(len(X), -1)


def build_master_hidden(family: str, widths: List[int], in_features: int, side: int,
                        kernel: int, stride: int, rng) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], Dict]:
    """Return per-path (W_hidden, b_hidden), nested as prefixes of a master."""
    max_w = max(widths)
    paths = []
    if family == "mlp":
        W_master = _sparse_int_weights(rng, (max_w, in_features))
        b_master = rng.integers(-4, 5, size=max_w, dtype=np.int32)
        for w in widths:
            paths.append((W_master[:w].copy(), b_master[:w].copy()))
        info = {"family": "mlp-random-polynomial-features"}
    elif family == "conv":
        filters = _sparse_int_weights(rng, (max_w, kernel, kernel))
        fbias = rng.integers(-4, 5, size=max_w, dtype=np.int32)
        out = conv_output_side(side, kernel, stride)
        for nf in widths:
            W = lower_conv_to_dense(filters[:nf], side, stride)
            b = np.repeat(fbias[:nf], out * out).astype(np.int32)
            paths.append((W, b))
        info = {
            "family": "conv-random-polynomial-features (lowered to exact dense LINEAR)",
            "kernel": kernel, "stride": stride, "conv_out_side": out,
            "filters_master": filters.tolist(), "filter_bias_master": fbias.tolist(),
        }
    else:
        raise ValueError(f"unknown family {family}")
    return paths, info


# --------------------------------------------------------------------------
# Exact arithmetic
# --------------------------------------------------------------------------

def hidden_features(X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    z = X.astype(np.int64) @ W.astype(np.int64).T + b.astype(np.int64)
    return z * z + 64 * z + 64


def hidden_bound(W: np.ndarray, b: np.ndarray, x_max: int) -> int:
    """Worst-case |z| for inputs in [0, x_max]; proves int64 safety of H."""
    zmax = int(np.abs(W.astype(np.int64)).sum(axis=1).max()) * x_max + int(np.abs(b).max())
    return zmax * zmax + 64 * zmax + 64


def train_output_layer(H_train, y_train):
    scaler = StandardScaler()
    Hs = scaler.fit_transform(H_train.astype(np.float64))
    clf = LogisticRegression(max_iter=5000, C=10.0, random_state=2026)
    clf.fit(Hs, y_train)
    coef_raw = clf.coef_ / scaler.scale_[None, :]
    bias_raw = clf.intercept_ - (clf.coef_ * scaler.mean_[None, :] / scaler.scale_[None, :]).sum(axis=1)
    return coef_raw, bias_raw


def integerize_output(coef_raw, bias_raw, H_all: np.ndarray, target_max_coef: float):
    """Scale V/c to integers with int32 parameters and float64-exact logits."""
    max_c = float(np.max(np.abs(coef_raw))) or 1.0
    max_b = float(np.max(np.abs(bias_raw))) or 1.0
    scale = min(target_max_coef / max_c, 1_000_000_000.0 / max_b)
    for _ in range(40):
        W_out = np.rint(coef_raw * scale).astype(np.int64)
        b_out = np.rint(bias_raw * scale).astype(np.int64)
        if np.max(np.abs(W_out)) > INT32_MAX or np.max(np.abs(b_out)) > INT32_MAX:
            scale /= 2.0
            continue
        # Exact Python-int bound on every score over all splits.
        row_abs = np.abs(H_all).max(axis=0).astype(object)
        bound = max(
            sum(int(abs(int(w))) * int(r) for w, r in zip(W_out[k], row_abs)) + abs(int(b_out[k]))
            for k in range(W_out.shape[0])
        )
        if bound < FLOAT64_EXACT_LIMIT:
            return W_out.astype(np.int32), b_out.astype(np.int32), float(scale), int(bound)
        scale /= 2.0
    raise RuntimeError("could not find an integer output scale with float64-exact logits")


def logits_exact(X, W_hidden, b_hidden, W_out, b_out):
    H = hidden_features(X, W_hidden, b_hidden)
    scores = H.astype(np.int64) @ W_out.astype(np.int64).T + b_out.astype(np.int64)
    if np.max(np.abs(scores)) >= FLOAT64_EXACT_LIMIT:
        raise RuntimeError("score magnitude exceeds float64-exact storage")
    return scores.astype(np.int64)


def write_arch(path: Path, in_features: int, width: int, n_classes: int) -> None:
    payload = {
        "layers": [
            {"type": "linear", "in_features": in_features, "out_features": width},
            {"type": "relu", "in_features": width, "out_features": width},
            {"type": "linear", "in_features": width, "out_features": n_classes},
            {"type": "softmax", "in_features": n_classes, "out_features": n_classes},
        ]
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset", choices=["digits", "mnist", "fashion_mnist"], default="mnist")
    ap.add_argument("--family", choices=["mlp", "conv"], default="mlp")
    ap.add_argument("--widths", default="16,48,96,160",
                    help="hidden widths (mlp) or filter counts (conv); strictly increasing")
    ap.add_argument("--pool", type=int, default=1, help="average-pool factor before quantization")
    ap.add_argument("--levels", type=int, default=16, help="input integer range is [0, levels]")
    ap.add_argument("--kernel", type=int, default=5)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--max-train", type=int, default=20000)
    ap.add_argument("--max-cal", type=int, default=5000)
    ap.add_argument("--max-test", type=int, default=5000)
    ap.add_argument("--target-max-coef", type=float, default=100_000.0)
    ap.add_argument("--data-home", default=None, help="OpenML cache directory")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    widths = [int(x) for x in args.widths.split(",") if x.strip()]
    if not widths or any(a >= b for a, b in zip(widths, widths[1:])):
        raise ValueError("widths must be a non-empty strictly increasing list")
    if not 1 <= args.levels <= 255:
        raise ValueError("levels must be in [1, 255]")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    (X_train, y_train, X_cal, y_cal, X_test, y_test), side, source = load_dataset(
        args.dataset, pool=args.pool, levels=args.levels, max_train=args.max_train,
        max_cal=args.max_cal, max_test=args.max_test, data_home=args.data_home)
    in_features = side * side
    n_classes = int(max(y_train.max(), y_cal.max(), y_test.max()) + 1)

    rng = np.random.default_rng(DATA_SEED)
    hidden_paths, family_info = build_master_hidden(
        args.family, widths, in_features, side, args.kernel, args.stride, rng)

    workload_id = (
        f"{args.dataset}{'-pool' + str(args.pool) if args.pool > 1 else ''}-"
        f"{args.family}-int{args.levels}-pcani-fixed-model-public-input-b1"
    )
    report = {
        "benchmark": f"{args.dataset} exact-integer PCANI ({args.family})",
        "workload_id": workload_id,
        "dataset": {
            "name": args.dataset, "source": source, "image_side": side,
            "input_features": in_features, "input_range": [0, args.levels],
            "pool": args.pool,
            "train": int(len(X_train)), "calibration": int(len(X_cal)), "test": int(len(X_test)),
        },
        "arithmetic": "int32 parameters / integer input / BN254 field embedding; no float quantization",
        "activation": "h=z^2+64z+64",
        "seed": DATA_SEED,
        "family": family_info,
        "paths": [],
    }

    cal_payload = {"labels": y_cal}
    test_payload = {"labels": y_test}
    models_dir = out / "models"
    models_dir.mkdir(exist_ok=True)
    X_all = np.concatenate([X_train, X_cal, X_test])

    for width_arg, (W_h, b_h) in zip(widths, hidden_paths):
        name = f"p{width_arg}"
        hidden_width = int(W_h.shape[0])
        if hidden_bound(W_h, b_h, args.levels) >= 2**62:
            raise RuntimeError(f"{name}: worst-case hidden activation could overflow int64")
        H_train = hidden_features(X_train, W_h, b_h)
        coef_raw, bias_raw = train_output_layer(H_train, y_train)
        H_all = hidden_features(X_all, W_h, b_h)
        W_o, b_o, fit_scale, score_bound = integerize_output(coef_raw, bias_raw, H_all, args.target_max_coef)

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
        write_arch(arch_path, in_features, hidden_width, n_classes)

        cal_payload[f"{name}_logits"] = cal_logits.astype(np.float64)
        test_payload[f"{name}_logits"] = test_logits.astype(np.float64)
        report["paths"].append({
            "name": name,
            "width": width_arg,
            "hidden_width": hidden_width,
            "calibration_accuracy": cal_acc,
            "test_accuracy": test_acc,
            "model_sha256": sha256_file(model_path),
            "fit_integer_scale": fit_scale,
            "observed_score_bound": score_bound,
            "parameter_count": int(model_values.size),
            "nonzero_hidden_weights": int(np.count_nonzero(W_h)),
        })
        print(f"[prepare] {name}: hidden={hidden_width} cal_acc={cal_acc:.4f} test_acc={test_acc:.4f}", flush=True)

    np.savez_compressed(out / "calibration.npz", **cal_payload)
    np.savez_compressed(out / "test.npz", **test_payload)
    np.save(out / "X_test.npy", X_test.astype(np.float32))
    np.save(out / "y_test.npy", y_test)
    (out / "prepare_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary = {k: v for k, v in report.items() if k != "family"}
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
