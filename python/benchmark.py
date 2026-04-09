#!/usr/bin/env python3
"""
CUDA-zkML benchmark runner.

This script measures the local CUDA demo flow and local CPU inference baselines.
Optional EZKL and Orion rows can be injected by dropping JSON files under
benchmarks/baselines/.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from quantize import quantize_symmetric
from zkml.model import Model


def run_json_script(script_name: str, extra_args: Optional[List[str]] = None) -> Dict[str, Any]:
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    cmd = [sys.executable, script_path]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    if not result.stdout.strip():
        raise RuntimeError(result.stderr.strip() or f"{script_name} produced no output")

    stdout = result.stdout.strip()
    json_candidate = stdout
    if not stdout.startswith("{"):
        first = stdout.find("{")
        last = stdout.rfind("}")
        if first != -1 and last != -1 and last > first:
            json_candidate = stdout[first:last + 1]

    try:
        return json.loads(json_candidate)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"{script_name} did not emit valid JSON.\nstdout={result.stdout[:500]}\nstderr={result.stderr[:500]}"
        ) from exc


def resolve_binary(build_path: str, binary_name: str) -> Optional[str]:
    candidates = [
        os.path.join(build_path, binary_name),
        os.path.join(build_path, binary_name + ".exe"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def find_cuda_zkml() -> Optional[str]:
    for path in [
        "../build",
        "../build/Release",
        "../cmake-build-release",
        "../cmake-build-release/Release",
        "../cmake-build-release2",
        "../cmake-build-release2/Release",
    ]:
        full = os.path.normpath(os.path.join(os.path.dirname(__file__), path))
        if resolve_binary(full, "zkml-prove"):
            return full
    return None


def parse_metric(stdout: str, label: str) -> Optional[float]:
    pattern = re.compile(rf"{re.escape(label)}\s*:\s*([0-9]+(?:\.[0-9]+)?)")
    match = pattern.search(stdout)
    if not match:
        return None
    return float(match.group(1))


def parse_proof_size(stdout: str) -> Optional[int]:
    match = re.search(r"Proof size\s*:\s*([0-9]+)\s*bytes", stdout)
    return int(match.group(1)) if match else None


def parse_prove_ms_fallback(stdout: str) -> Optional[float]:
    match = re.search(r"\[Prove\]\s+Proof generated in\s+([0-9]+(?:\.[0-9]+)?)\s*ms", stdout)
    return float(match.group(1)) if match else None


def parse_verify_ms_fallback(stdout: str) -> Optional[float]:
    matches = re.findall(r"\[Verify\]\s+Verification PASSED in\s+([0-9]+(?:\.[0-9]+)?)\s*ms", stdout)
    return float(matches[0]) if matches else None


def parse_saved_proof_size(stdout: str) -> Optional[int]:
    match = re.search(r"\[IO\]\s+Saved proof to .*?\(([0-9]+)\s+bytes\)", stdout)
    return int(match.group(1)) if match else None


def _workspace_tempdir() -> str:
    candidates = [
        os.environ.get("ZKML_TMPDIR"),
        os.path.join(os.path.dirname(__file__), "..", ".zkml_tmp"),
        os.path.join(os.path.dirname(__file__), "..", "build"),
        os.getcwd(),
        tempfile.gettempdir(),
    ]
    for base in candidates:
        if not base:
            continue
        try:
            os.makedirs(base, exist_ok=True)
            tmpdir = os.path.join(base, f"zkml_bench_{uuid.uuid4().hex}")
            os.makedirs(tmpdir, exist_ok=False)
            probe = os.path.join(tmpdir, "probe.tmp")
            with open(probe, "w", encoding="utf-8") as f:
                f.write("ok")
            os.remove(probe)
            return tmpdir
        except (OSError, PermissionError):
            continue
    raise RuntimeError("Could not create a writable temporary directory for benchmarks")


def benchmark_cuda_zkml(build_path: str, n_runs: int = 3) -> Dict[str, Any]:
    prove_cmd = resolve_binary(build_path, "zkml-prove")
    if not prove_cmd:
        return {"system": "CUDA-zkML (GTX 1650)", "status": "unavailable", "note": "zkml-prove not found"}

    times: List[Dict[str, Any]] = []
    print(f"\n[1] CUDA-zkML demo benchmark ({prove_cmd})")

    for run_idx in range(n_runs):
        tmpdir = _workspace_tempdir()
        try:
            start = time.time()
            result = subprocess.run(
                [prove_cmd, "--demo"],
                capture_output=True,
                text=True,
                timeout=600,
                cwd=tmpdir,
            )
            wall_s = time.time() - start
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if result.returncode != 0 or "Proof valid:        YES" not in result.stdout:
            print(f"  Run {run_idx + 1}: FAILED")
            stderr = result.stderr.strip()
            if stderr:
                print(stderr[:500])
            continue

        run = {
            "inference_ms": parse_metric(result.stdout, "Inference time"),
            "setup_ms": parse_metric(result.stdout, "Setup time"),
            "prove_ms": parse_metric(result.stdout, "Proving time") or parse_prove_ms_fallback(result.stdout),
            "verify_ms": parse_metric(result.stdout, "Verification time") or parse_verify_ms_fallback(result.stdout),
            "proof_size_bytes": parse_proof_size(result.stdout) or parse_saved_proof_size(result.stdout),
            "wall_s": wall_s,
        }
        if None in (run["prove_ms"], run["verify_ms"], run["proof_size_bytes"]):
            print(f"  Run {run_idx + 1}: FAILED (could not parse prove/verify timings)")
            continue
        times.append(run)
        print(
            f"  Run {run_idx + 1}: "
            f"infer={_format_value(run['inference_ms'])}ms "
            f"setup={_format_value(run['setup_ms'])}ms "
            f"prove={run['prove_ms']:.2f}ms "
            f"verify={run['verify_ms']:.2f}ms"
        )

    if not times:
        return {"system": "CUDA-zkML (GTX 1650)", "status": "error", "note": "all demo benchmark runs failed"}

    return {
        "system": "CUDA-zkML (GTX 1650)",
        "model": "MNIST MLP (784→128→10)",
        "status": "measured",
        "avg_inference_ms": float(np.mean([t["inference_ms"] for t in times])),
        "avg_setup_ms": float(np.mean([t["setup_ms"] for t in times])),
        "avg_prove_ms": float(np.mean([t["prove_ms"] for t in times])),
        "avg_verify_ms": float(np.mean([t["verify_ms"] for t in times])),
        "avg_total_s": float(np.mean([t["wall_s"] for t in times])),
        "proof_size_bytes": int(times[0]["proof_size_bytes"]),
        "n_runs": len(times),
        "hardware": "NVIDIA GTX 1650 (4GB, SM 7.5)",
        "note": "Measured from zkml-prove --demo on the local machine",
    }


def benchmark_tiny_transformer(build_path: str, n_runs: int = 1) -> Dict[str, Any]:
    prove_cmd = resolve_binary(build_path, "zkml-prove")
    if not prove_cmd:
        return {"system": "CUDA-zkML Tiny Transformer (GTX 1650)", "status": "unavailable", "note": "zkml-prove not found"}

    times: List[Dict[str, Any]] = []
    print(f"\n[2] CUDA-zkML tiny transformer benchmark ({prove_cmd})")

    for run_idx in range(n_runs):
        tmpdir = _workspace_tempdir()
        try:
            model = Model.create_tiny_transformer()
            model_path = os.path.join(tmpdir, "transformer.bin")
            arch_path = model_path + ".arch.json"
            input_path = os.path.join(tmpdir, "input.bin")
            proof_path = os.path.join(tmpdir, "proof.bin")
            vk_path = os.path.join(tmpdir, "vk.bin")
            pi_path = os.path.join(tmpdir, "public_inputs.bin")

            model.to_binary(model_path)
            model.save_architecture(arch_path)
            rng = np.random.RandomState(123 + run_idx)
            rng.randn(model.input_size).astype(np.float32).tofile(input_path)

            start = time.time()
            result = subprocess.run(
                [
                    prove_cmd,
                    "--model", model_path,
                    "--arch", arch_path,
                    "--input", input_path,
                    "--output", proof_path,
                    "--vk", vk_path,
                    "--public-inputs", pi_path,
                ],
                capture_output=True,
                text=True,
                timeout=600,
            )
            wall_s = time.time() - start
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        if result.returncode != 0 or "Proof valid: YES" not in result.stdout:
            print(f"  Run {run_idx + 1}: FAILED")
            stderr = result.stderr.strip()
            if stderr:
                print(stderr[:500])
            continue

        run = {
            "inference_ms": parse_metric(result.stdout, "Inference time"),
            "setup_ms": parse_metric(result.stdout, "Setup time"),
            "prove_ms": parse_metric(result.stdout, "Proving time"),
            "verify_ms": parse_metric(result.stdout, "Verification time"),
            "proof_size_bytes": parse_proof_size(result.stdout),
            "wall_s": wall_s,
        }
        if None in (
            run["inference_ms"],
            run["setup_ms"],
            run["prove_ms"],
            run["verify_ms"],
            run["proof_size_bytes"],
        ):
            print(f"  Run {run_idx + 1}: FAILED (could not parse timings)")
            continue
        times.append(run)
        print(
            f"  Run {run_idx + 1}: "
            f"infer={run['inference_ms']:.2f}ms "
            f"setup={run['setup_ms']:.2f}ms "
            f"prove={run['prove_ms']:.2f}ms "
            f"verify={run['verify_ms']:.2f}ms"
        )

    if not times:
        return {
            "system": "CUDA-zkML Tiny Transformer (GTX 1650)",
            "model": "Tiny transformer (4 tokens × 8 hidden, single-head attention)",
            "status": "error",
            "note": "all tiny transformer benchmark runs failed",
        }

    return {
        "system": "CUDA-zkML Tiny Transformer (GTX 1650)",
        "model": "Tiny transformer (4 tokens × 8 hidden, single-head attention)",
        "status": "measured",
        "avg_inference_ms": float(np.mean([t["inference_ms"] for t in times])),
        "avg_setup_ms": float(np.mean([t["setup_ms"] for t in times])),
        "avg_prove_ms": float(np.mean([t["prove_ms"] for t in times])),
        "avg_verify_ms": float(np.mean([t["verify_ms"] for t in times])),
        "avg_total_s": float(np.mean([t["wall_s"] for t in times])),
        "proof_size_bytes": int(times[0]["proof_size_bytes"]),
        "n_runs": len(times),
        "hardware": "NVIDIA GTX 1650 (4GB, SM 7.5)",
        "note": "Measured from zkml-prove on the tiny transformer attention path",
    }


def _run_float_mlp(model: Model, x: np.ndarray) -> np.ndarray:
    return model.float_inference(x)


def _run_quantized_field_mlp(model: Model, x: np.ndarray) -> np.ndarray:
    values = x.astype(np.float32).copy()

    for layer in model.layers:
        if layer.layer_type == "linear":
            w_q, w_scale = quantize_symmetric(layer.weights, bits=8)
            x_q, x_scale = quantize_symmetric(values, bits=8)
            y_q = w_q.astype(np.int64) @ x_q.astype(np.int64)
            values = y_q.astype(np.float32) / (w_scale * x_scale)
            if layer.bias is not None:
                values = values + layer.bias
        elif layer.layer_type == "relu":
            values = np.maximum(values, 0)
        elif layer.layer_type == "softmax":
            shifted = values - np.max(values)
            exp = np.exp(shifted)
            values = exp / np.sum(exp)
    return values


def benchmark_cpu_baselines(n_runs: int = 5) -> List[Dict[str, Any]]:
    model = Model.create_mnist_mlp()
    rng = np.random.RandomState(42)

    float_times_ms: List[float] = []
    quant_times_ms: List[float] = []
    agreement = 0

    print("\n[2] CPU inference baselines")
    for run_idx in range(n_runs):
        x = rng.randn(784).astype(np.float32)

        start = time.perf_counter()
        float_out = _run_float_mlp(model, x)
        float_times_ms.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        quant_out = _run_quantized_field_mlp(model, x)
        quant_times_ms.append((time.perf_counter() - start) * 1000.0)

        if int(np.argmax(float_out)) == int(np.argmax(quant_out)):
            agreement += 1

        print(
            f"  Run {run_idx + 1}: "
            f"float={float_times_ms[-1]:.2f}ms "
            f"quantized-field={quant_times_ms[-1]:.2f}ms"
        )

    agreement_pct = 100.0 * agreement / n_runs
    return [
        {
            "system": "CPU Float32 Inference",
            "model": "MNIST MLP (784→128→10)",
            "status": "measured",
            "avg_inference_ms": float(np.mean(float_times_ms)),
            "avg_setup_ms": None,
            "avg_prove_ms": None,
            "avg_verify_ms": None,
            "avg_total_s": float(np.mean(float_times_ms) / 1000.0),
            "proof_size_bytes": 0,
            "n_runs": n_runs,
            "hardware": platform.processor() or "CPU",
            "note": "Reference float32 forward pass only",
        },
        {
            "system": "CPU Quantized Field Inference",
            "model": "MNIST MLP (784→128→10)",
            "status": "measured",
            "avg_inference_ms": float(np.mean(quant_times_ms)),
            "avg_setup_ms": None,
            "avg_prove_ms": None,
            "avg_verify_ms": None,
            "avg_total_s": float(np.mean(quant_times_ms) / 1000.0),
            "proof_size_bytes": 0,
            "n_runs": n_runs,
            "hardware": platform.processor() or "CPU",
            "note": f"Inference-only CPU baseline; float/quantized top-1 agreement {agreement_pct:.1f}%",
        },
    ]


def _baseline_candidates(name: str) -> List[str]:
    root = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "benchmarks", "baselines"))
    return [
        os.path.join(root, f"{name}.json"),
        os.path.join(root, f"{name}_results.json"),
        os.path.join(root, f"{name}.benchmark.json"),
    ]


def load_external_baseline(name: str, display_name: str) -> Dict[str, Any]:
    for candidate in _baseline_candidates(name):
        if os.path.exists(candidate):
            with open(candidate, "r", encoding="utf-8") as f:
                data = json.load(f)
            data.setdefault("system", display_name)
            data.setdefault("status", "measured")
            data.setdefault("note", f"Loaded from {os.path.basename(candidate)}")
            return data

    return {
        "system": display_name,
        "status": "unavailable",
        "avg_inference_ms": None,
        "avg_setup_ms": None,
        "avg_prove_ms": None,
        "avg_verify_ms": None,
        "avg_total_s": None,
        "proof_size_bytes": None,
        "hardware": None,
        "note": f"No baseline JSON found under benchmarks/baselines/{name}.json",
    }


def benchmark_live_ezkl() -> Dict[str, Any]:
    print("  EZKL: running live benchmark")
    try:
        row = run_json_script("run_ezkl_benchmark.py")
        row.setdefault("system", "EZKL")
        return row
    except Exception as exc:
        return {
            "system": "EZKL",
            "status": "error",
            "avg_inference_ms": None,
            "avg_setup_ms": None,
            "avg_prove_ms": None,
            "avg_verify_ms": None,
            "avg_total_s": None,
            "proof_size_bytes": None,
            "hardware": "CPU",
            "note": f"Live EZKL runner failed: {exc}",
        }


def benchmark_live_orion() -> Dict[str, Any]:
    print("  Orion: running live benchmark probe")
    try:
        row = run_json_script("run_orion_benchmark.py")
        row.setdefault("system", "Orion")
        return row
    except Exception as exc:
        return {
            "system": "Orion",
            "status": "error",
            "avg_inference_ms": None,
            "avg_setup_ms": None,
            "avg_prove_ms": None,
            "avg_verify_ms": None,
            "avg_total_s": None,
            "proof_size_bytes": None,
            "hardware": "CPU",
            "note": f"Live Orion runner failed: {exc}",
        }


def _format_value(value: Optional[float], precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{precision}f}"


def format_results_markdown(results: List[Dict[str, Any]]) -> str:
    lines = [
        "# CUDA-zkML Benchmark Results",
        "",
        "| System | Status | Inference (ms) | Setup (ms) | Prove (ms) | Verify (ms) | Proof Size | Hardware | Note |",
        "|--------|--------|----------------|------------|------------|-------------|------------|----------|------|",
    ]

    for row in results:
        proof_size = row.get("proof_size_bytes")
        proof_size_str = "N/A" if proof_size is None else f"{proof_size} B"
        hardware = row.get("hardware") or "N/A"
        note = row.get("note", "")
        lines.append(
            "| {system} | {status} | {infer} | {setup} | {prove} | {verify} | {size} | {hardware} | {note} |".format(
                system=row.get("system", "Unknown"),
                status=row.get("status", "unknown"),
                infer=_format_value(row.get("avg_inference_ms")),
                setup=_format_value(row.get("avg_setup_ms")),
                prove=_format_value(row.get("avg_prove_ms")),
                verify=_format_value(row.get("avg_verify_ms")),
                size=proof_size_str,
                hardware=hardware,
                note=note,
            )
        )

    lines.extend(
        [
            "",
            "*Rows include the MNIST demo, tiny transformer attention path, local CPU baselines, and optional external comparisons.*",
            f"*Date: {time.strftime('%Y-%m-%d')}*",
        ]
    )
    return "\n".join(lines) + "\n"


def safe_print(text: str) -> None:
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = sys.stdout.encoding or "utf-8"
        sys.stdout.buffer.write(text.encode(encoding, errors="replace"))
        sys.stdout.buffer.write(b"\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run CUDA-zkML benchmarks")
    parser.add_argument("--runs", type=int, default=3, help="Number of measured runs")
    parser.add_argument("--live-ezkl", action="store_true", help="Run the local EZKL benchmark helper")
    parser.add_argument("--live-orion", action="store_true", help="Run the local Orion benchmark helper/probe")
    args = parser.parse_args()

    print("=" * 60)
    print("CUDA-zkML Benchmark Suite")
    print("=" * 60)

    results: List[Dict[str, Any]] = []

    build_path = find_cuda_zkml()
    if build_path:
        results.append(benchmark_cuda_zkml(build_path, n_runs=args.runs))
        results.append(benchmark_tiny_transformer(build_path, n_runs=max(1, min(args.runs, 2))))
    else:
        print("\n[1] CUDA-zkML: NOT FOUND (build the project first)")
        results.append(
            {
                "system": "CUDA-zkML (GTX 1650)",
                "status": "unavailable",
                "note": "Build directory not found",
            }
        )
        results.append(
            {
                "system": "CUDA-zkML Tiny Transformer (GTX 1650)",
                "status": "unavailable",
                "note": "Build directory not found",
            }
        )

    results.extend(benchmark_cpu_baselines(n_runs=max(args.runs, 3)))

    print("\n[4] External baselines")
    ezkl = benchmark_live_ezkl() if args.live_ezkl else load_external_baseline("ezkl", "EZKL")
    orion = benchmark_live_orion() if args.live_orion else load_external_baseline("orion", "Orion")
    print(f"  EZKL: {ezkl['status']}")
    print(f"  Orion: {orion['status']}")
    results.append(ezkl)
    results.append(orion)

    payload = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host": platform.node(),
        "runs": args.runs,
        "results": results,
    }

    results_dir = os.path.join(os.path.dirname(__file__), "..", "benchmarks", "results")
    os.makedirs(results_dir, exist_ok=True)

    json_path = os.path.join(results_dir, "benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    markdown = format_results_markdown(results)
    md_path = os.path.join(results_dir, "benchmark_results.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(markdown)

    print("\nResults written to:")
    print(f"  {json_path}")
    print(f"  {md_path}")
    safe_print("\n" + markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
