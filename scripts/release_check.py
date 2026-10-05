#!/usr/bin/env python3
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    p = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout[-4000:], "stderr": p.stderr[-4000:]}


def main():
    checks = []
    checks.append(run([sys.executable, "-m", "compileall", "-q", "python", "tests"]))
    checks.append(run([sys.executable, "-m", "pytest", "-q", "tests"]))

    src = (ROOT / "src/prover/groth16.cu").read_text(encoding="utf-8")
    save = src[src.index("bool ProvingKey::save"):src.index("bool ProvingKey::load_streaming")]
    static = {
        "pk_writer_has_no_debug_trapdoor_write": "write_fr(debug_trapdoor" not in save,
        "pk_writer_has_no_scalar_query_write": "write_vec_fr(A_query_scalars" not in save,
        "pk_v3_hardened_marker": "Saved hardened v3 proving key" in save,
        "naive_aggregation_disabled": "[Aggregate] DISABLED" in src,
        "rs_delta_correction_present": "Standard Groth16 correction term" in src,
    }
    cuda = {
        "nvcc": shutil.which("nvcc"),
        "nvidia_smi": shutil.which("nvidia-smi"),
        "cuda_validation_possible_here": bool(shutil.which("nvcc") and shutil.which("nvidia-smi")),
    }
    report = {
        "status": "pass" if all(c["returncode"] == 0 for c in checks) and all(static.values()) else "fail",
        "python_checks": checks,
        "static_security_checks": static,
        "cuda_environment": cuda,
        "note": "CUDA/Groth16 runtime validation requires an NVIDIA CUDA environment when cuda_validation_possible_here=false.",
    }
    out = ROOT / "results/release_validation_2026.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "pass" else 1

if __name__ == "__main__":
    raise SystemExit(main())
