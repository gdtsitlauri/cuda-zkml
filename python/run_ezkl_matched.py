#!/usr/bin/env python3
"""Matched-workload EZKL comparison for a prepared PCANI benchmark.

Unlike ``run_ezkl_benchmark.py`` (a tiny 4->4->2 reference model), this script
proves *the same exact-integer models* that CUDA-zkML proves, so rows can share
a ``workload_id`` under benchmarks/COMPARABILITY_PROTOCOL.md:

* identical integer weights and activation h = z^2 + 64 z + 64;
* public input, fixed (circuit-constant) parameters, public outputs;
* EZKL input/param scale forced to 0, i.e. no float quantization;
* EZKL's public outputs are checked against the exact Python logits and the
  row is marked ``comparable=false`` on any mismatch or forced rescaling.

Every stage is timed separately; setup and SRS generation are reported but are
not part of ``prove_ms``.  Failures are recorded, never hidden.
"""
from __future__ import annotations

import argparse
import asyncio
import inspect
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

F32_EXACT = 2**24
BIAS_SPLIT = 2**20


def _call(fn, *args, **kwargs):
    """Call an EZKL function that may be sync or async depending on version."""
    out = fn(*args, **kwargs)
    if inspect.isawaitable(out):
        out = asyncio.run(out)
    return out


def load_integer_model(model_path: Path) -> List[Dict[str, Any]]:
    arch = json.loads(Path(str(model_path) + ".arch.json").read_text(encoding="utf-8"))
    vals = np.frombuffer(model_path.read_bytes(), dtype="<i4").astype(np.int64)
    layers, off = [], 0
    for spec in arch["layers"]:
        if spec["type"] != "linear":
            layers.append({"type": spec["type"]})
            continue
        n_in, n_out = int(spec["in_features"]), int(spec["out_features"])
        W = vals[off: off + n_in * n_out].reshape(n_out, n_in)
        off += n_in * n_out
        b = vals[off: off + n_out]
        off += n_out
        layers.append({"type": "linear", "W": W, "b": b})
    if off != len(vals):
        raise ValueError(f"{model_path}: parameter count does not match architecture")
    return layers


def build_onnx(layers: List[Dict[str, Any]], in_features: int, path: Path) -> int:
    """Export the exact integer model (z^2+64z+64 activation) to ONNX."""
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    nodes, inits = [], []
    cur, n_out = "input", in_features
    lin_idx = 0
    for layer in layers:
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            if np.max(np.abs(W)) >= F32_EXACT:
                raise ValueError("weights must be exactly representable in float32")
            # ONNX constants are float32: split the int32 bias into two exactly
            # representable parts, b = b_lo + b_hi * 2**20, added in the circuit.
            b_hi = np.trunc(b / BIAS_SPLIT).astype(np.int64)
            b_lo = b - b_hi * BIAS_SPLIT
            w_name, b_name, out = f"W{lin_idx}", f"b{lin_idx}", f"lin{lin_idx}"
            inits.append(numpy_helper.from_array(W.astype(np.float32), w_name))
            inits.append(numpy_helper.from_array(b_lo.astype(np.float32), b_name))
            nodes.append(helper.make_node("Gemm", [cur, w_name, b_name], [out], transB=1))
            if np.any(b_hi):
                hi_name = f"bhi{lin_idx}"
                inits.append(numpy_helper.from_array((b_hi * BIAS_SPLIT).astype(np.float32), hi_name))
                nodes.append(helper.make_node("Add", [out, hi_name], [f"{out}_b"]))
                out = f"{out}_b"
            cur, n_out = out, int(W.shape[0])
            lin_idx += 1
        elif layer["type"] == "relu":
            # Same polynomial as CUDA-zkML's integer RELU_APPROX layer.
            k = f"act{lin_idx}"
            c64 = f"{k}_c64"
            inits.append(numpy_helper.from_array(np.array([64.0], dtype=np.float32), c64))
            nodes += [
                helper.make_node("Mul", [cur, cur], [f"{k}_sq"]),
                helper.make_node("Add", [cur, c64], [f"{k}_zp64"]),
                helper.make_node("Mul", [f"{k}_zp64", c64], [f"{k}_64z"]),
                # (z + 64) * 64 = 64 z + 4096, so subtract 4032 to obtain 64 z + 64.
                helper.make_node("Add", [f"{k}_sq", f"{k}_64z"], [f"{k}_sum"]),
            ]
            off = f"{k}_off"
            inits.append(numpy_helper.from_array(np.array([-4032.0], dtype=np.float32), off))
            nodes.append(helper.make_node("Add", [f"{k}_sum", off], [k]))
            cur = k
        # softmax is outside the proved statement, as in CUDA-zkML.
    nodes.append(helper.make_node("Identity", [cur], ["output"]))
    graph = helper.make_graph(
        nodes, "pcani_exact_integer",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, in_features])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, n_out])],
        inits,
    )
    model = helper.make_model(graph, producer_name="cuda-zkml-pcani", opset_imports=[helper.make_opsetid("", 13)])
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    return n_out


def _felt_outputs(witness: Dict[str, Any], scale: int) -> List[int]:
    import ezkl
    outs = witness["outputs"][0]
    res = []
    for felt in outs:
        if hasattr(ezkl, "felt_to_int"):
            res.append(int(ezkl.felt_to_int(felt)))
        else:
            res.append(int(round(float(ezkl.felt_to_float(felt, scale)))))
    return res


def decomposition_legs(value_bound: int, base_bits: int = 14) -> int:
    """EZKL range-decomposes values in base 2**14; enough legs for |value| (+sign)."""
    return max(2, -(-(int(value_bound).bit_length() + 1) // base_bits))


def run_path(name: str, model_path: Path, sample: np.ndarray, expected: List[int],
             workdir: Path, repeats: int, value_bound: int) -> Dict[str, Any]:
    import ezkl

    workdir.mkdir(parents=True, exist_ok=True)
    onnx_path = workdir / "model.onnx"
    data_path = workdir / "input.json"
    settings_path = workdir / "settings.json"
    compiled_path = workdir / "network.compiled"
    witness_path = workdir / "witness.json"
    srs_path = workdir / "kzg.srs"
    vk_path = workdir / "vk.key"
    pk_path = workdir / "pk.key"
    stage: Dict[str, float] = {}
    row: Dict[str, Any] = {"path": name, "status": "error", "comparable": False}

    def timed(key, fn, *a, **k):
        t0 = time.perf_counter()
        out = _call(fn, *a, **k)
        stage[key] = (time.perf_counter() - t0) * 1000.0
        if out is False:
            raise RuntimeError(f"{key} returned False")
        return out

    try:
        layers = load_integer_model(model_path)
        build_onnx(layers, int(sample.size), onnx_path)
        data_path.write_text(json.dumps({"input_data": [[float(v) for v in sample.tolist()]]}), encoding="utf-8")

        run_args = ezkl.PyRunArgs()
        run_args.input_visibility = "public"
        run_args.param_visibility = "fixed"
        run_args.output_visibility = "public"
        run_args.input_scale = 0
        run_args.param_scale = 0
        # Exact integer scores far exceed EZKL's default 2-leg decomposition.
        legs = decomposition_legs(value_bound)
        if hasattr(run_args, "decomp_legs"):
            run_args.decomp_legs = legs
        row["decomp_legs"] = legs
        timed("settings_ms", ezkl.gen_settings, str(onnx_path), str(settings_path), py_run_args=run_args)
        # Only resolve logrows; scales stay pinned to 0 (exact integers).
        timed("calibrate_ms", ezkl.calibrate_settings, str(data_path), str(onnx_path),
              str(settings_path), "resources", scales=[0])
        settings = json.loads(settings_path.read_text(encoding="utf-8"))
        ra = settings.get("run_args", {})
        scales = {"input_scale": ra.get("input_scale"), "param_scale": ra.get("param_scale")}
        logrows = int(ra.get("logrows") or settings.get("logrows") or 17)
        row.update({"logrows": logrows, "ezkl_scales": scales})

        timed("compile_ms", ezkl.compile_circuit, str(onnx_path), str(compiled_path), str(settings_path))
        timed("witness_ms", ezkl.gen_witness, str(data_path), str(compiled_path), str(witness_path))
        witness = json.loads(witness_path.read_text(encoding="utf-8"))
        got = _felt_outputs(witness, int(scales["input_scale"] or 0))
        outputs_match = got == expected
        row["outputs_match_exact_logits"] = outputs_match

        timed("srs_ms", ezkl.gen_srs, str(srs_path), logrows)
        timed("setup_ms", ezkl.setup, str(compiled_path), str(vk_path), str(pk_path), srs_path=str(srs_path))

        prove_ms, verify_ms, sizes = [], [], []
        for r in range(repeats):
            proof_path = workdir / f"proof_{r:02d}.json"
            t0 = time.perf_counter()
            _call(ezkl.prove, str(witness_path), str(compiled_path), str(pk_path),
                  proof_path=str(proof_path), srs_path=str(srs_path))
            prove_ms.append((time.perf_counter() - t0) * 1000.0)
            t0 = time.perf_counter()
            ok = _call(ezkl.verify, str(proof_path), str(settings_path), str(vk_path), srs_path=str(srs_path))
            verify_ms.append((time.perf_counter() - t0) * 1000.0)
            if not ok:
                raise RuntimeError(f"EZKL verification failed on repeat {r}")
            sizes.append(proof_path.stat().st_size)

        row.update({
            "status": "measured",
            "comparable": bool(outputs_match and scales["input_scale"] == 0 and scales["param_scale"] == 0),
            "median_prove_ms": float(statistics.median(prove_ms)),
            "min_prove_ms": min(prove_ms), "max_prove_ms": max(prove_ms),
            "median_verify_ms": float(statistics.median(verify_ms)),
            "proof_size_bytes": int(statistics.median(sizes)),
            "raw_prove_ms": prove_ms, "raw_verify_ms": verify_ms,
            "repeats": repeats,
        })
    except BaseException as exc:  # record, do not hide
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["stage_ms"] = stage
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benchmark-dir", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--paths", default="", help="comma-separated subset of path names (default: all)")
    ap.add_argument("--workdir", default=".zkml_tmp/ezkl_matched")
    args = ap.parse_args()

    # EZKL's setup reads $HOME, which is not defined on Windows.
    os.environ.setdefault("HOME", str(Path.home()))

    bench = Path(args.benchmark_dir)
    prep =json.loads((bench / "prepare_report.json").read_text(encoding="utf-8"))
    X_test = np.load(bench / "X_test.npy")
    sample = np.asarray(X_test[args.sample_index])
    if not np.all(sample == np.rint(sample)):
        raise ValueError("benchmark inputs must be exact integers")
    wanted = {p for p in args.paths.split(",") if p}

    try:
        import ezkl
        ezkl_version = getattr(ezkl, "__version__", "unknown")
    except ImportError as exc:
        ezkl_version = None
        import_error = str(exc)

    rows = []
    with np.load(bench / "test.npz") as td:
        for p in prep["paths"]:
            name = p["name"]
            if wanted and name not in wanted:
                continue
            expected = [int(v) for v in td[f"{name}_logits"][args.sample_index]]
            if ezkl_version is None:
                rows.append({"path": name, "status": "unavailable", "comparable": False,
                             "error": f"EZKL not installed: {import_error}"})
                continue
            print(f"[ezkl] proving {name} ...", flush=True)
            # Bound on every value in the graph: the largest of the observed score
            # bound (pcani_prepare.py) and this sample's exact intermediates.
            layers = load_integer_model(bench / "models" / name / "model.i32.bin")
            z = layers[0]["W"] @ sample.astype(np.int64) + layers[0]["b"]
            bound = max(int(p.get("observed_score_bound", 0)), int(np.abs(z * z + 64 * z + 64).max()),
                        max(abs(v) for v in expected))
            rows.append(run_path(name, bench / "models" / name / "model.i32.bin",
                                 sample, expected, Path(args.workdir) / name, args.repeats, bound))
            print(json.dumps({k: v for k, v in rows[-1].items() if not k.startswith("raw_")}), flush=True)

    payload = {
        "system": "EZKL",
        "ezkl_version": ezkl_version,
        "workload_id": prep.get("workload_id", "digits-exact-integer"),
        "benchmark": prep.get("benchmark"),
        "statement": "public input / fixed exact-integer model / public pre-softmax scores",
        "hardware": f"{platform.processor() or platform.machine()} (EZKL CPU prover)",
        "python": sys.version,
        "sample_index": args.sample_index,
        "note": "Cross-hardware if CUDA-zkML ran on a GPU; report as such.",
        "rows": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0 if all(r.get("status") == "measured" for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
