import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "python"))

onnx = pytest.importorskip("onnx")
from onnx.reference import ReferenceEvaluator  # noqa: E402

import pcani_prepare  # noqa: E402
import run_ezkl_matched  # noqa: E402


def test_onnx_export_reproduces_exact_integer_model(tmp_path):
    bench = tmp_path / "bench"
    pcani_prepare.main(["--out", str(bench), "--dataset", "digits", "--family", "mlp", "--widths", "4,8"])
    X = np.load(bench / "X_test.npy")[:8]
    with np.load(bench / "test.npz") as td:
        expected = td["p4_logits"][:8]
    layers = run_ezkl_matched.load_integer_model(bench / "models" / "p4" / "model.i32.bin")
    onnx_path = tmp_path / "p4.onnx"
    n_out = run_ezkl_matched.build_onnx(layers, X.shape[1], onnx_path)
    assert n_out == 10
    sess = ReferenceEvaluator(str(onnx_path))
    for row, exp in zip(X, expected):
        (got,) = sess.run(None, {"input": row.reshape(1, -1).astype(np.float32)})
        # ONNX float32 cannot hold the large exact scores; EZKL evaluates the
        # same graph exactly in the field.  Check the graph algebra closely.
        np.testing.assert_allclose(got.reshape(-1), exp, rtol=1e-5, atol=1.0)
        assert int(np.argmax(got)) == int(np.argmax(exp))
