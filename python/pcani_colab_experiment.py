#!/usr/bin/env python3
"""End-to-end NVIDIA validation for PCANI protocols v1/v2.

Protocol v1 reproduces the cumulative proof-chain baseline. Protocol v2 is the
default and emits one selected-path route certificate per representative sample.
Both modes generate a real sklearn-Digits exact-integer benchmark, create
model-specific Groth16 keys, check exact CUDA/Python public outputs, and measure
repeated prover/verification timings. No benchmark number is hard-coded. A failed
correctness check aborts the run.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from typing import Dict, List, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from zkml.adaptive import PathSpec, route_indices  # noqa: E402
from zkml.artifacts import load_public_inputs  # noqa: E402
from zkml.pcani_protocol import (  # noqa: E402
    ManifestPath,
    ProtocolManifest,
    fr_to_signed,
    make_bundle,
    sha256_file,
    verify_route_bundle,
)
from zkml.pcani_protocol_v2 import (  # noqa: E402
    make_single_proof_certificate,
    verify_single_proof_certificate,
)


_METRIC_PATTERNS = {
    "inference_ms": re.compile(r"Inference time:\s*([0-9.]+)\s*ms"),
    "setup_ms": re.compile(r"Setup time:\s*([0-9.]+)\s*ms"),
    "prove_ms": re.compile(r"Proving time:\s*([0-9.]+)\s*ms"),
    "verify_ms": re.compile(r"Verification time:\s*([0-9.]+)\s*ms"),
    "proof_size_bytes": re.compile(r"Proof size:\s*([0-9]+)\s*bytes"),
    "model_tag64": re.compile(r"Quantized model tag:\s*(0x[0-9a-fA-F]+)"),
}


def run(cmd: Sequence[str], *, cwd: Path = ROOT, log: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    print("+", " ".join(map(str, cmd)), flush=True)
    proc = subprocess.run(list(map(str, cmd)), cwd=str(cwd), text=True, capture_output=True)
    text = (proc.stdout or "") + (proc.stderr or "")
    print(text, end="" if text.endswith("\n") else "\n")
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(text, encoding="utf-8")
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(map(str, cmd))}")
    return proc


def capture(cmd: Sequence[str]) -> str:
    try:
        return subprocess.check_output(list(map(str, cmd)), text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return f"unavailable: {exc}"


def parse_metrics(text: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for name, pat in _METRIC_PATTERNS.items():
        m = pat.search(text)
        if m:
            if name == "model_tag64":
                out[name] = m.group(1).lower()
            elif name == "proof_size_bytes":
                out[name] = int(m.group(1))
            else:
                out[name] = float(m.group(1))
    return out


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def save_sample(path: Path, row: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(row, dtype=np.float32))


def load_reference_logits(npz_path: Path, name: str, index: int) -> List[int]:
    with np.load(npz_path) as data:
        vals = np.asarray(data[f"{name}_logits"][index], dtype=np.float64)
    if not np.all(vals == np.rint(vals)):
        raise RuntimeError(f"reference logits for {name} are not exact integers")
    return [int(x) for x in vals]


def statement_outputs(public_path: Path, meta_path: Path) -> List[int]:
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    nout = int(meta["public_output_count"])
    public = load_public_inputs(str(public_path))
    return [fr_to_signed(x) for x in public[:nout]]


def prove_cmd(
    prove_bin: Path,
    *, model: Path,
    sample: Path,
    proof: Path,
    vk: Path,
    public_inputs: Path,
    statement_meta: Path,
    pk_save: Path | None = None,
    pk_load: Path | None = None,
) -> List[str]:
    cmd = [
        str(prove_bin),
        "--model", str(model),
        "--input", str(sample),
        "--output", str(proof),
        "--vk", str(vk),
        "--public-inputs", str(public_inputs),
        "--pcani-statement",
        "--statement-meta", str(statement_meta),
        "--integer-model",
        "--integer-input",
    ]
    if pk_save is not None:
        cmd += ["--pk-save", str(pk_save)]
    if pk_load is not None:
        cmd += ["--pk-load", str(pk_load)]
    return cmd


def choose_representatives(chosen: np.ndarray, max_bundles: int) -> List[int]:
    observed = sorted(set(int(x) for x in chosen.tolist()))
    if len(observed) > max_bundles:
        # Preserve earliest and full routes, spread any remaining choices.
        keep = [observed[0], observed[-1]]
        for x in observed[1:-1]:
            if len(keep) >= max_bundles:
                break
            keep.insert(-1, x)
        observed = sorted(set(keep[:max_bundles]))
    return [int(np.flatnonzero(chosen == ridx)[0]) for ridx in observed]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--benchmark-dir", default="benchmarks/pcani_digits")
    ap.add_argument("--results-dir", default="results/pcani_colab_final")
    ap.add_argument("--widths", default="16,48,96,160")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-accuracy-drop", type=float, default=0.0, help="calibration-set accuracy-loss budget")
    ap.add_argument("--heldout-accuracy-drop-limit", type=float, default=0.01, help="predeclared held-out acceptance tolerance; never used for calibration")
    ap.add_argument("--max-route-bundles", type=int, default=4)
    ap.add_argument("--protocol-version", type=int, choices=[1, 2], default=2)
    ap.add_argument(
        "--prepare-script", default="pcani_digits_prepare.py",
        help="benchmark generator in python/; pcani_prepare.py supports MNIST and conv families",
    )
    ap.add_argument(
        "--prepare-args", default="",
        help="extra arguments for the prepare script, e.g. '--dataset mnist --family conv'",
    )
    ap.add_argument("--archive-name", default="PCANI_final_results", help="result ZIP basename in the repo root")
    args = ap.parse_args()
    if args.repeats < 3:
        raise ValueError("use at least 3 timing repeats")

    build = (ROOT / args.build_dir).resolve()
    bench = (ROOT / args.benchmark_dir).resolve()
    results = (ROOT / args.results_dir).resolve()
    prove_bin = build / "zkml-prove"
    verify_bin = build / "zkml-verify"
    for exe in (prove_bin, verify_bin):
        if not exe.exists():
            raise FileNotFoundError(f"missing executable: {exe}")

    if results.exists():
        shutil.rmtree(results)
    results.mkdir(parents=True)
    logs = results / "logs"
    artifacts = results / "artifacts"
    artifacts.mkdir()

    env = {
        "timestamp_unix": time.time(),
        "python": sys.version,
        "platform": sys.platform,
        "nvidia_smi": capture(["nvidia-smi"]),
        "nvcc_version": capture(["nvcc", "--version"]),
        "cmake_version": capture(["cmake", "--version"]),
        "git_commit": capture(["git", "rev-parse", "HEAD"]),
        "benchmark_build": "Release / ZKML_FAST_DEBUG_LOOP=OFF required",
    }
    (results / "environment.json").write_text(json.dumps(env, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Real dataset preparation is deterministic and independent of proof timing.
    if bench.exists():
        shutil.rmtree(bench)
    run([
        sys.executable, str(HERE / args.prepare_script),
        "--out", str(bench), "--widths", args.widths,
        *shlex.split(args.prepare_args),
    ], log=logs / "prepare_benchmark.log")
    prep = json.loads((bench / "prepare_report.json").read_text(encoding="utf-8"))
    path_names = [str(p["name"]) for p in prep["paths"]]

    X_test = np.load(bench / "X_test.npy")
    setup_index = 0
    setup_sample = results / "setup_sample.npy"
    save_sample(setup_sample, X_test[setup_index])

    proof_costs: Dict[str, Dict[str, object]] = {}
    path_state: Dict[str, Dict[str, object]] = {}

    for pinfo in prep["paths"]:
        name = str(pinfo["name"])
        pdir = artifacts / name
        pdir.mkdir(parents=True)
        model = bench / "models" / name / "model.i32.bin"
        arch = Path(str(model) + ".arch.json")
        if not arch.exists():
            raise FileNotFoundError(arch)

        pk = pdir / "pk.bin"
        vk = pdir / "vk.bin"
        proof = pdir / "setup.proof.bin"
        pub = pdir / "setup.public.bin"
        meta = pdir / "statement.json"
        cmd = prove_cmd(
            prove_bin, model=model, sample=setup_sample, proof=proof, vk=vk,
            public_inputs=pub, statement_meta=meta, pk_save=pk,
        )
        proc = run(cmd, log=logs / f"{name}_setup.log")
        setup_metrics = parse_metrics((proc.stdout or "") + (proc.stderr or ""))
        required = {"setup_ms", "prove_ms", "verify_ms", "proof_size_bytes", "model_tag64"}
        if not required.issubset(setup_metrics):
            raise RuntimeError(f"missing setup metrics for {name}: {required - set(setup_metrics)}")

        # Independent CLI verification of the persisted artifact.
        run([
            str(verify_bin), "--vk", str(vk), "--proof", str(proof),
            "--public-inputs", str(pub),
        ], log=logs / f"{name}_native_verify.log")

        ref = load_reference_logits(bench / "test.npz", name, setup_index)
        got = statement_outputs(pub, meta)
        if got != ref:
            raise RuntimeError(
                f"exact CUDA/Python output mismatch for {name}:\nexpected={ref}\nactual={got}"
            )

        repeat_metrics: List[Dict[str, object]] = []
        for r in range(args.repeats):
            rproof = pdir / f"repeat_{r:02d}.proof.bin"
            rpub = pdir / f"repeat_{r:02d}.public.bin"
            rmeta = pdir / f"repeat_{r:02d}.statement.json"
            rcmd = prove_cmd(
                prove_bin, model=model, sample=setup_sample, proof=rproof, vk=vk,
                public_inputs=rpub, statement_meta=rmeta, pk_load=pk,
            )
            rp = run(rcmd, log=logs / f"{name}_repeat_{r:02d}.log")
            met = parse_metrics((rp.stdout or "") + (rp.stderr or ""))
            for key in ("prove_ms", "verify_ms", "inference_ms", "proof_size_bytes"):
                if key not in met:
                    raise RuntimeError(f"missing {key} in repeat {r} for {name}")
            if statement_outputs(rpub, rmeta) != ref:
                raise RuntimeError(f"repeat output mismatch for {name}, repeat {r}")
            repeat_metrics.append(met)

        prove_times = [float(m["prove_ms"]) for m in repeat_metrics]
        verify_times = [float(m["verify_ms"]) for m in repeat_metrics]
        inference_times = [float(m["inference_ms"]) for m in repeat_metrics]
        statement = json.loads(meta.read_text(encoding="utf-8"))
        stats = {
            "median_prove_ms": float(statistics.median(prove_times)),
            "p95_prove_ms": percentile(prove_times, 95),
            "min_prove_ms": min(prove_times),
            "max_prove_ms": max(prove_times),
            "median_verify_ms": float(statistics.median(verify_times)),
            "median_inference_ms": float(statistics.median(inference_times)),
            "setup_ms": float(setup_metrics["setup_ms"]),
            "proof_size_bytes": int(setup_metrics["proof_size_bytes"]),
            "r1cs_constraints": int(statement["r1cs_constraints"]),
            "r1cs_variables": int(statement["r1cs_variables"]),
            "model_tag64": str(statement["model_tag64"]).lower(),
            "model_sha256": sha256_file(str(model)),
            "vk_sha256": sha256_file(str(vk)),
            "repeats": args.repeats,
            "raw_repeats": repeat_metrics,
            "cuda_python_exact_output_match": True,
        }
        proof_costs[name] = stats
        path_state[name] = {
            "model": str(model), "pk": str(pk), "vk": str(vk), "meta": str(meta), **stats,
        }
        (pdir / "metrics.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Candidate path ordering is part of the algorithm.  Do not silently massage
    # noisy timings into monotonic costs: require the measured medians to agree.
    medians = [float(proof_costs[n]["median_prove_ms"]) for n in path_names]
    if any(medians[i] > medians[i + 1] for i in range(len(medians) - 1)):
        raise RuntimeError(
            "measured prover costs are not non-decreasing across candidate paths; "
            "rerun with more repeats or inspect GPU noise. costs=" + repr(dict(zip(path_names, medians)))
        )

    proof_cost_path = results / "proof_costs.json"
    proof_cost_path.write_text(json.dumps(proof_costs, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Calibrate on held-out calibration data, evaluate once on held-out test data.
    cost_semantics = "proof_chain" if args.protocol_version == 1 else "selected_proof"
    run([
        sys.executable, str(HERE / "pcani_digits_finalize.py"),
        "--benchmark-dir", str(bench),
        "--proof-costs", str(proof_cost_path),
        "--max-accuracy-drop", str(args.max_accuracy_drop),
        "--cost-semantics", cost_semantics,
    ], log=logs / "finalize_policy.log")
    policy = json.loads((bench / "policy.json").read_text(encoding="utf-8"))
    final_results = json.loads((bench / "final_results.json").read_text(encoding="utf-8"))
    shutil.copy2(bench / "policy.json", results / "policy.json")
    shutil.copy2(bench / "final_results.json", results / "final_results.json")

    thresholds = [float(x) for x in policy["thresholds"]]
    policy_digest = str(policy["policy_digest"])

    manifest_paths: List[ManifestPath] = []
    for i, pinfo in enumerate(prep["paths"]):
        name = str(pinfo["name"])
        threshold = None if i == len(prep["paths"]) - 1 else thresholds[i]
        state = path_state[name]
        manifest_paths.append(ManifestPath(
            name=name,
            threshold=threshold,
            vk_sha256=str(state["vk_sha256"]),
            model_tag64=str(state["model_tag64"]),
            model_sha256=str(state["model_sha256"]),
            standalone_proof_cost_ms=float(state["median_prove_ms"]),
        ))
    manifest = ProtocolManifest(
        policy_digest=policy_digest,
        confidence_kind="margin",
        paths=tuple(manifest_paths),
        protocol_version=args.protocol_version,
        statement_mode=(
            "pcani-fixed-model/public-input" if args.protocol_version == 1 else
            "pcani-fixed-model/public-input/single-selected-proof"
        ),
    )
    manifest_path = results / "protocol_manifest.json"
    manifest.save(str(manifest_path))

    # Choose representative held-out samples for each route observed by the policy.
    with np.load(bench / "test.npz") as td:
        y_test = np.asarray(td["labels"], dtype=np.int64)
        test_logits = {n: np.asarray(td[f"{n}_logits"], dtype=np.float64) for n in path_names}
    path_specs = [PathSpec(n, float(proof_costs[n]["median_prove_ms"])) for n in path_names]
    chosen = route_indices(test_logits, path_specs, thresholds, confidence_kind="margin")
    representative_indices = choose_representatives(chosen, args.max_route_bundles)

    bundle_reports: List[Dict[str, object]] = []
    bundles_root = results / ("proof_chain_bundles" if args.protocol_version == 1 else "single_proof_certificates")
    bundles_root.mkdir()
    for sample_idx in representative_indices:
        chosen_idx = int(chosen[sample_idx])
        chosen_name = path_names[chosen_idx]
        bdir = bundles_root / f"sample_{sample_idx:03d}_{chosen_name}"
        bdir.mkdir()
        sample_file = bdir / "input.npy"
        save_sample(sample_file, X_test[sample_idx])

        if args.protocol_version == 1:
            attempts: List[Mapping[str, str]] = []
            path_indices = range(chosen_idx + 1)
        else:
            attempts = []
            path_indices = [chosen_idx]

        for i in path_indices:
            name = path_names[i]
            state = path_state[name]
            model = Path(str(state["model"]))
            proof = bdir / f"{name}.proof.bin"
            public = bdir / f"{name}.public.bin"
            meta = bdir / f"{name}.statement.json"
            cmd = prove_cmd(
                prove_bin, model=model, sample=sample_file, proof=proof,
                vk=Path(str(state["vk"])), public_inputs=public,
                statement_meta=meta, pk_load=Path(str(state["pk"])),
            )
            run(cmd, log=logs / f"bundle_{sample_idx:03d}_{name}.log")
            ref = load_reference_logits(bench / "test.npz", name, sample_idx)
            if statement_outputs(public, meta) != ref:
                raise RuntimeError(f"bundle exact-output mismatch sample={sample_idx} path={name}")
            attempts.append({
                "path_name": name,
                "proof": str(proof),
                "vk": str(state["vk"]),
                "public_inputs": str(public),
                "statement_meta": str(meta),
            })

        if args.protocol_version == 1:
            bundle_path = bdir / "bundle.json"
            make_bundle(
                str(bundle_path), policy_digest=policy_digest,
                chosen_path=chosen_name, attempts=attempts,
            )
            report = verify_route_bundle(str(bundle_path), str(manifest_path), str(verify_bin))
            report_payload = report.to_dict()
            valid = report.valid
            errors = report.errors
        else:
            bundle_path = bdir / "certificate.json"
            make_single_proof_certificate(
                str(bundle_path), policy_digest=policy_digest,
                chosen_path=chosen_name, attempt=attempts[0],
            )
            report = verify_single_proof_certificate(str(bundle_path), str(manifest_path), str(verify_bin))
            report_payload = report.to_dict()
            valid = report.valid
            errors = report.errors

        report_payload.update({
            "sample_index": sample_idx,
            "label": int(y_test[sample_idx]),
            "policy_route_index": chosen_idx,
        })
        (bdir / "verification_report.json").write_text(
            json.dumps(report_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        if not valid:
            raise RuntimeError(f"route verification failed for sample {sample_idx}: {errors}")
        bundle_reports.append(report_payload)

    summary = {
        "status": "completed_gpu_validation_run",
        "benchmark": prep,
        "proof_costs": proof_costs,
        "pcani": final_results,
        "protocol_manifest_sha256": sha256_file(str(manifest_path)),
        "protocol_version": args.protocol_version,
        "route_evidence_reports": bundle_reports,
        "correctness_gates": {
            "native_groth16_verify": True,
            "cuda_python_exact_logits": True,
            "model_specific_vk_pinned": True,
            "same_public_input_cross_proof_linkage": (args.protocol_version == 1),
            "selected_path_acceptance_verified_from_proof_bound_public_outputs": True,
        },
        "privacy_scope": {
            "input": f"public in protocol v{args.protocol_version}",
            "model": "exactly bound by fixed circuit/model-specific key; model-hiding is not claimed",
            "routing": "public threshold decision over proof-bound scores; not a hidden in-circuit comparison",
        },
        "predeclared_criteria": {
            "calibration_max_accuracy_drop": args.max_accuracy_drop,
            "heldout_max_accuracy_drop": args.heldout_accuracy_drop_limit,
            "require_positive_measured_proof_cost_savings": True,
        },
    }
    test_result = final_results["test"]
    summary["hypothesis_supported_on_this_run"] = bool(
        float(test_result["accuracy_drop"]) <= args.heldout_accuracy_drop_limit + 1e-12
        and float(test_result["proof_cost_savings"]) > 0.0
    )
    (results / "run_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Include benchmark data/model artifacts in the returned result archive so a
    # later audit can reproduce every reported policy decision.
    packaged = results / "benchmark_snapshot"
    if packaged.exists():
        shutil.rmtree(packaged)
    shutil.copytree(bench, packaged)

    archive_base = ROOT / args.archive_name
    zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=str(results)))
    print("\n=== PCANI GPU VALIDATION COMPLETE ===")
    print(json.dumps({
        "zip": str(zip_path),
        "adaptive_test_accuracy": final_results["test"]["accuracy"],
        "full_path_accuracy": final_results["test"]["full_path_accuracy"],
        "proof_cost_savings": final_results["test"]["proof_cost_savings"],
        "route_counts": final_results["test"]["route_counts"],
        "verified_route_evidence": len(bundle_reports),
        "protocol_version": args.protocol_version,
        "hypothesis_supported_on_this_run": summary["hypothesis_supported_on_this_run"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
