import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PYDIR = HERE.parent / "python"
sys.path.insert(0, str(PYDIR))

import pcani_prepare  # noqa: E402


def _reference_logits(model_path: Path, arch_path: Path, X: np.ndarray) -> np.ndarray:
    """Independent pure-Python-int evaluation of an exported integer model."""
    arch = json.loads(arch_path.read_text())
    lin = [l for l in arch["layers"] if l["type"] == "linear"]
    vals = np.frombuffer(model_path.read_bytes(), dtype="<i4").astype(np.int64).tolist()
    off = 0
    params = []
    for l in lin:
        n_w = l["in_features"] * l["out_features"]
        W = [vals[off + r * l["in_features"]: off + (r + 1) * l["in_features"]] for r in range(l["out_features"])]
        off += n_w
        b = vals[off: off + l["out_features"]]
        off += l["out_features"]
        params.append((W, b))
    assert off == len(vals)
    (W1, b1), (W2, b2) = params
    out = []
    for row in X.astype(np.int64).tolist():
        z = [sum(w * x for w, x in zip(Wr, row)) + br for Wr, br in zip(W1, b1)]
        h = [v * v + 64 * v + 64 for v in z]
        out.append([sum(w * x for w, x in zip(Wr, h)) + br for Wr, br in zip(W2, b2)])
    return np.array(out, dtype=object)


def _check_exact(bench: Path, n: int = 12):
    report = json.loads((bench / "prepare_report.json").read_text())
    X = np.load(bench / "X_test.npy")[:n]
    assert np.all(X == np.rint(X))
    with np.load(bench / "test.npz") as td:
        for p in report["paths"]:
            stored = td[f"{p['name']}_logits"][:n]
            model = bench / "models" / p["name"] / "model.i32.bin"
            ref = _reference_logits(model, Path(str(model) + ".arch.json"), X)
            assert [[int(v) for v in r] for r in stored] == ref.tolist(), p["name"]
    return report


def test_conv_lowering_matches_direct_convolution():
    rng = np.random.default_rng(0)
    filters = rng.integers(-2, 3, size=(3, 3, 3)).astype(np.int32)
    X = rng.integers(0, 17, size=(5, 64)).astype(np.int32)
    dense = pcani_prepare.lower_conv_to_dense(filters, side=8, stride=2)
    assert dense.shape == (3 * 3 * 3, 64)
    np.testing.assert_array_equal(
        X.astype(np.int64) @ dense.T.astype(np.int64),
        pcani_prepare.direct_conv(X, filters, side=8, stride=2),
    )


def test_digits_mlp_reproduces_original_t4_benchmark_bytes(tmp_path):
    """The generalised script must regenerate the completed T4 Digits models."""
    old = tmp_path / "old"
    new = tmp_path / "new"
    subprocess.run([sys.executable, str(PYDIR / "pcani_digits_prepare.py"), "--out", str(old),
                    "--widths", "16,48,96,160"], check=True, capture_output=True)
    pcani_prepare.main(["--out", str(new), "--dataset", "digits", "--family", "mlp",
                        "--widths", "16,48,96,160"])
    for w in (16, 48, 96, 160):
        a = (old / "models" / f"p{w}" / "model.i32.bin").read_bytes()
        b = (new / "models" / f"p{w}" / "model.i32.bin").read_bytes()
        assert a == b, f"p{w} differs from the original Digits benchmark"
    with np.load(old / "test.npz") as ta, np.load(new / "test.npz") as tb:
        for w in (16, 48, 96, 160):
            np.testing.assert_array_equal(ta[f"p{w}_logits"], tb[f"p{w}_logits"])
    _check_exact(new)


def test_digits_conv_family_is_exact_and_nested(tmp_path):
    out = tmp_path / "conv"
    pcani_prepare.main(["--out", str(out), "--dataset", "digits", "--family", "conv",
                        "--widths", "2,4,8", "--kernel", "3", "--stride", "1"])
    report = _check_exact(out)
    assert report["workload_id"].startswith("digits-conv-int16")
    hidden = [p["hidden_width"] for p in report["paths"]]
    assert hidden == [2 * 36, 4 * 36, 8 * 36]
    # Nested paths: the smaller path's hidden rows are a prefix of the larger.
    small = np.frombuffer((out / "models" / "p2" / "model.i32.bin").read_bytes(), dtype="<i4")
    large = np.frombuffer((out / "models" / "p8" / "model.i32.bin").read_bytes(), dtype="<i4")
    n_small_hidden = 72 * 64
    np.testing.assert_array_equal(small[:n_small_hidden], large[:n_small_hidden])
    for p in report["paths"]:
        assert p["observed_score_bound"] < 2**52
        assert p["test_accuracy"] > 0.5


def test_finalize_accepts_generalised_benchmark(tmp_path):
    out = tmp_path / "bench"
    pcani_prepare.main(["--out", str(out), "--dataset", "digits", "--family", "conv",
                        "--widths", "2,4", "--kernel", "3", "--stride", "1"])
    costs = tmp_path / "costs.json"
    costs.write_text(json.dumps({"p2": {"median_prove_ms": 10.0}, "p4": {"median_prove_ms": 20.0}}))
    env = dict(os.environ, PYTHONPATH=str(PYDIR))
    subprocess.run([sys.executable, str(PYDIR / "pcani_digits_finalize.py"),
                    "--benchmark-dir", str(out), "--proof-costs", str(costs),
                    "--max-accuracy-drop", "0.0"], check=True, capture_output=True, env=env)
    final = json.loads((out / "final_results.json").read_text())
    assert final["benchmark"] == "digits exact-integer PCANI (conv)"
    assert final["workload_id"].startswith("digits-conv")
