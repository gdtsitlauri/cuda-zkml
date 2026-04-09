#!/usr/bin/env python3
"""
Run a live EZKL benchmark on a tiny reference MLP and emit a benchmark row as JSON.

This is intended to provide a reproducible local comparison path on machines where
the EZKL toolchain is installed, while preserving the exact failure mode when a
Windows wheel or runtime issue prevents full proving.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import uuid
from typing import Any, Dict

import numpy as np


def repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def workspace_tempdir() -> str:
    candidates = [
        os.environ.get("ZKML_TMPDIR"),
        os.path.join(repo_root(), ".zkml_tmp"),
        os.getcwd(),
        tempfile.gettempdir(),
    ]
    for base in candidates:
        if not base:
            continue
        try:
            os.makedirs(base, exist_ok=True)
            probe_dir = os.path.join(base, f"ezkl_probe_{uuid.uuid4().hex}")
            os.makedirs(probe_dir, exist_ok=False)
            try:
                probe_path = os.path.join(probe_dir, "probe.tmp")
                with open(probe_path, "w", encoding="utf-8") as f:
                    f.write("ok")
            finally:
                shutil.rmtree(probe_dir, ignore_errors=True)
            live_dir = os.path.join(base, f"ezkl_live_{uuid.uuid4().hex}")
            os.makedirs(live_dir, exist_ok=False)
            return live_dir
        except (OSError, PermissionError):
            continue
    raise RuntimeError("Could not create a writable temporary directory for EZKL")


def build_reference_onnx(path: str) -> None:
    import onnx
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.RandomState(7)
    w1 = (rng.randn(4, 4).astype(np.float32) * 0.1)
    b1 = np.zeros(4, dtype=np.float32)
    w2 = (rng.randn(2, 4).astype(np.float32) * 0.1)
    b2 = np.zeros(2, dtype=np.float32)

    input_info = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])
    output_info = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])

    graph = helper.make_graph(
        [
            helper.make_node("Gemm", ["input", "w1", "b1"], ["hidden"], alpha=1.0, beta=1.0, transB=1),
            helper.make_node("Relu", ["hidden"], ["hidden_relu"]),
            helper.make_node("Gemm", ["hidden_relu", "w2", "b2"], ["output"], alpha=1.0, beta=1.0, transB=1),
        ],
        "ezkl_reference_mlp",
        [input_info],
        [output_info],
        [
            numpy_helper.from_array(w1, "w1"),
            numpy_helper.from_array(b1, "b1"),
            numpy_helper.from_array(w2, "w2"),
            numpy_helper.from_array(b2, "b2"),
        ],
    )

    model = helper.make_model(
        graph,
        producer_name="cuda-zkml-benchmark",
        opset_imports=[helper.make_opsetid("", 13)],
    )
    onnx.save(model, path)


def run_ezkl_live() -> Dict[str, Any]:
    try:
        import ezkl
    except ImportError as exc:
        return {
            "system": "EZKL",
            "status": "unavailable",
            "avg_inference_ms": None,
            "avg_setup_ms": None,
            "avg_prove_ms": None,
            "avg_verify_ms": None,
            "avg_total_s": None,
            "proof_size_bytes": None,
            "hardware": "CPU",
            "note": f"EZKL not installed: {exc}",
        }

    tmpdir = workspace_tempdir()
    stage_ms: Dict[str, float] = {}
    started = time.perf_counter()

    try:
        model_path = os.path.join(tmpdir, "model.onnx")
        data_path = os.path.join(tmpdir, "input.json")
        settings_path = os.path.join(tmpdir, "settings.json")
        compiled_path = os.path.join(tmpdir, "network.compiled")
        witness_path = os.path.join(tmpdir, "witness.json")
        srs_path = os.path.join(tmpdir, "kzg.srs")
        vk_path = os.path.join(tmpdir, "vk.key")
        pk_path = os.path.join(tmpdir, "pk.key")
        proof_path = os.path.join(tmpdir, "proof.json")

        build_reference_onnx(model_path)

        t0 = time.perf_counter()
        ezkl.gen_random_data(model_path, data_path, [], 42)
        stage_ms["input_ms"] = (time.perf_counter() - t0) * 1000.0

        t0 = time.perf_counter()
        ok = ezkl.gen_settings(model_path, settings_path)
        stage_ms["settings_ms"] = (time.perf_counter() - t0) * 1000.0
        if not ok:
            raise RuntimeError("gen_settings returned false")

        t0 = time.perf_counter()
        ok = ezkl.calibrate_settings(data_path, model_path, settings_path, "resources")
        stage_ms["calibrate_ms"] = (time.perf_counter() - t0) * 1000.0
        if not ok:
            raise RuntimeError("calibrate_settings returned false")

        t0 = time.perf_counter()
        ok = ezkl.compile_circuit(model_path, compiled_path, settings_path)
        stage_ms["compile_ms"] = (time.perf_counter() - t0) * 1000.0
        if not ok:
            raise RuntimeError("compile_circuit returned false")

        t0 = time.perf_counter()
        ok = ezkl.gen_witness(data_path, compiled_path, witness_path)
        stage_ms["witness_ms"] = (time.perf_counter() - t0) * 1000.0
        if not ok:
            raise RuntimeError("gen_witness returned false")

        with open(settings_path, "r", encoding="utf-8") as f:
            settings = json.load(f)
        logrows = int(settings.get("run_args", {}).get("logrows") or settings.get("logrows") or 17)

        t0 = time.perf_counter()
        ok = ezkl.gen_srs(srs_path, logrows)
        stage_ms["srs_ms"] = (time.perf_counter() - t0) * 1000.0
        if ok is False:
            raise RuntimeError("gen_srs returned false")

        t0 = time.perf_counter()
        ok = ezkl.setup(compiled_path, vk_path, pk_path, srs_path=srs_path, witness_path=witness_path)
        stage_ms["setup_ms"] = (time.perf_counter() - t0) * 1000.0
        if not ok:
            raise RuntimeError("setup returned false")

        t0 = time.perf_counter()
        ok = ezkl.prove(witness_path, compiled_path, pk_path, proof_path=proof_path, srs_path=srs_path)
        stage_ms["prove_ms"] = (time.perf_counter() - t0) * 1000.0
        if not ok:
            raise RuntimeError("prove returned false")

        t0 = time.perf_counter()
        ok = ezkl.verify(proof_path, settings_path, vk_path, srs_path=srs_path)
        stage_ms["verify_ms"] = (time.perf_counter() - t0) * 1000.0
        if not ok:
            raise RuntimeError("verify returned false")

        proof_size = os.path.getsize(proof_path) if os.path.exists(proof_path) else None
        total_s = time.perf_counter() - started
        return {
            "system": "EZKL",
            "model": "Tiny reference MLP (4→4→2)",
            "status": "measured",
            "avg_inference_ms": stage_ms.get("witness_ms"),
            "avg_setup_ms": stage_ms.get("setup_ms"),
            "avg_prove_ms": stage_ms.get("prove_ms"),
            "avg_verify_ms": stage_ms.get("verify_ms"),
            "avg_total_s": total_s,
            "proof_size_bytes": proof_size,
            "hardware": "CPU",
            "note": "Live local EZKL run on a tiny reference ONNX model",
            "stage_ms": stage_ms,
        }
    except BaseException as exc:
        total_s = time.perf_counter() - started
        return {
            "system": "EZKL",
            "model": "Tiny reference MLP (4→4→2)",
            "status": "error",
            "avg_inference_ms": stage_ms.get("witness_ms"),
            "avg_setup_ms": stage_ms.get("setup_ms"),
            "avg_prove_ms": stage_ms.get("prove_ms"),
            "avg_verify_ms": stage_ms.get("verify_ms"),
            "avg_total_s": total_s,
            "proof_size_bytes": os.path.getsize(proof_path) if 'proof_path' in locals() and os.path.exists(proof_path) else None,
            "hardware": "CPU",
            "note": f"Live EZKL benchmark failed: {exc}",
            "stage_ms": stage_ms,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a live EZKL benchmark and emit JSON")
    parser.add_argument("--output", default=None, help="Optional path to write the benchmark JSON row")
    args = parser.parse_args()

    row = run_ezkl_live()
    payload = json.dumps(row, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
    print(payload)
    return 0 if row.get("status") == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
