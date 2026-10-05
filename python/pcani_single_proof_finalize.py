#!/usr/bin/env python3
"""Re-analyze a completed PCANI GPU run under protocol-v2 single-proof semantics.

This does not invent new timings. It reuses the measured standalone prover and
inference medians from the completed NVIDIA run, recalibrates the deterministic
routing policy using the stored calibration snapshot, evaluates the held-out
snapshot, and emits a protocol-v2 manifest plus representative single-proof
route certificates derived from already verified proof-chain bundles.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from zkml.adaptive import PathSpec, calibrate_policy, evaluate_policy, route_indices
from zkml.pcani_protocol import ManifestPath, ProtocolManifest
from zkml.pcani_protocol_v2 import make_single_proof_certificate


def load_json(path: Path):
    return json.loads(path.read_text())


def margin(logits: np.ndarray) -> np.ndarray:
    part = np.partition(np.asarray(logits), -2, axis=1)
    return part[:, -1] - part[:, -2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    r = args.results_dir.resolve()
    out = (args.out or (r / "single_proof_v2")).resolve()
    out.mkdir(parents=True, exist_ok=True)

    costs = load_json(r / "proof_costs.json")
    v1_manifest = load_json(r / "protocol_manifest.json")
    cal = np.load(r / "benchmark_snapshot" / "calibration.npz")
    test = np.load(r / "benchmark_snapshot" / "test.npz")
    names = [p["name"] for p in v1_manifest["paths"]]
    cal_logits = {name: cal[f"{name}_logits"] for name in names}
    test_logits = {name: test[f"{name}_logits"] for name in names}
    cal_labels = cal["labels"]
    test_labels = test["labels"]

    paths = [
        PathSpec(
            name,
            float(costs[name]["median_prove_ms"]),
            workload_id="digits-exact-integer",
            model_digest=str(costs[name]["model_sha256"]),
        )
        for name in names
    ]

    policy = calibrate_policy(
        cal_logits,
        cal_labels,
        paths,
        max_accuracy_drop=0.0,
        confidence_kind="margin",
        cost_semantics="selected_proof",
    )
    ev = evaluate_policy(
        test_logits,
        test_labels,
        paths,
        policy.thresholds,
        confidence_kind="margin",
        cost_semantics="selected_proof",
    )
    chosen = route_indices(test_logits, paths, policy.thresholds, confidence_kind="margin")

    # Measured inference overhead for the sequential cheap-to-expensive policy.
    infer_ms = {name: float(costs[name]["median_inference_ms"]) for name in names}
    expected_infer = 0.0
    for idx, name in enumerate(names):
        count = int((chosen == idx).sum())
        expected_infer += count * sum(infer_ms[n] for n in names[: idx + 1])
    expected_infer /= len(test_labels)
    static_e2e = infer_ms[names[-1]] + float(costs[names[-1]]["median_prove_ms"])
    adaptive_e2e = expected_infer + ev.expected_proof_cost
    e2e_savings = 1.0 - adaptive_e2e / static_e2e

    # Rebuild a protocol-v2 manifest. Thresholds are pinned to the recalibrated policy.
    manifest_paths = []
    v1_by_name = {p["name"]: p for p in v1_manifest["paths"]}
    for idx, name in enumerate(names):
        old = v1_by_name[name]
        threshold = None if idx == len(names) - 1 else float(policy.thresholds[idx])
        manifest_paths.append(
            ManifestPath(
                name=name,
                threshold=threshold,
                vk_sha256=str(old["vk_sha256"]),
                model_tag64=str(old["model_tag64"]),
                model_sha256=str(old.get("model_sha256", "")),
                standalone_proof_cost_ms=float(costs[name]["median_prove_ms"]),
            )
        )
    manifest = ProtocolManifest(
        policy_digest=policy.digest(),
        confidence_kind="margin",
        paths=tuple(manifest_paths),
        protocol_version=2,
        statement_mode="pcani-fixed-model/public-input/single-selected-proof",
    )
    manifest_path = out / "protocol_v2_manifest.json"
    manifest.save(manifest_path)
    policy.save(out / "policy_v2.json")

    # Representative certificate evidence: reuse the chosen proof from each already
    # native-verified protocol-v1 bundle. The proof itself is unchanged.
    cert_dir = out / "representative_certificates"
    cert_dir.mkdir(exist_ok=True)
    representative = []
    for bundle_path in sorted((r / "proof_chain_bundles").glob("*/bundle.json")):
        bundle = load_json(bundle_path)
        chosen_name = bundle["chosen_path"]
        attempt = next(a for a in bundle["attempts"] if a["path_name"] == chosen_name)
        # Resolve old bundle-relative paths, then make v2 certificate relocatable.
        abs_attempt = dict(attempt)
        for key in ("proof", "vk", "public_inputs", "statement_meta"):
            abs_attempt[key] = str((bundle_path.parent / attempt[key]).resolve())
        cert_path = cert_dir / f"{bundle_path.parent.name}.json"
        make_single_proof_certificate(
            str(cert_path),
            policy_digest=policy.digest(),
            chosen_path=chosen_name,
            attempt=abs_attempt,
        )
        old_report = load_json(bundle_path.parent / "verification_report.json")
        idx = names.index(chosen_name)
        threshold = None if idx == len(names)-1 else float(policy.thresholds[idx])
        chosen_margin = int(old_report["route_margins"][chosen_name])
        representative.append({
            "sample": bundle_path.parent.name,
            "chosen_path": chosen_name,
            "native_groth16_verified_in_gpu_run": bool(old_report["native_verification"][chosen_name]),
            "chosen_margin": chosen_margin,
            "threshold": threshold,
            "selected_path_accepts": True if threshold is None else chosen_margin >= threshold,
            "source_v1_bundle_valid": bool(old_report["valid"]),
        })

    result = {
        "status": "protocol_v2_reanalysis_completed",
        "evidence_boundary": {
            "gpu_timings": "measured on the uploaded completed NVIDIA run; no timing is fabricated",
            "heldout_routing": "recomputed from stored exact-integer test logits",
            "single_proof_cost": "weighted mean of measured standalone per-path median prover times",
            "representative_crypto": "selected proofs were already native-Groth16 verified in protocol-v1 bundles; v2 reuses those same proofs",
            "limitation": "not every held-out sample was individually proved; GPU cost is estimated from repeated per-path measurements",
        },
        "protocol_v1_observed": {
            "expected_chain_proof_cost_ms": 2899.9466388888895,
            "static_full_proof_cost_ms": float(costs[names[-1]]["median_prove_ms"]),
            "proof_cost_savings": 1.0 - 2899.9466388888895 / float(costs[names[-1]]["median_prove_ms"]),
        },
        "protocol_v2_single_proof": {
            "calibration_accuracy": policy.calibration_accuracy,
            "calibration_full_path_accuracy": policy.full_path_accuracy,
            "thresholds": policy.thresholds,
            "heldout_accuracy": ev.accuracy,
            "heldout_full_path_accuracy": ev.full_path_accuracy,
            "heldout_accuracy_drop": ev.accuracy_drop,
            "route_counts": ev.route_counts,
            "expected_selected_proof_cost_ms": ev.expected_proof_cost,
            "static_full_proof_cost_ms": ev.full_path_cost,
            "proof_cost_savings": ev.proof_cost_savings,
            "proof_cost_speedup_x": ev.full_path_cost / ev.expected_proof_cost,
            "expected_sequential_inference_ms": expected_infer,
            "estimated_adaptive_inference_plus_proof_ms": adaptive_e2e,
            "static_full_inference_plus_proof_ms": static_e2e,
            "estimated_end_to_end_savings": e2e_savings,
            "estimated_end_to_end_speedup_x": static_e2e / adaptive_e2e,
        },
        "measured_path_costs": {
            name: {
                "median_prove_ms": float(costs[name]["median_prove_ms"]),
                "median_inference_ms": float(costs[name]["median_inference_ms"]),
                "median_verify_ms": float(costs[name]["median_verify_ms"]),
                "proof_size_bytes": int(costs[name]["proof_size_bytes"]),
                "r1cs_constraints": int(costs[name]["r1cs_constraints"]),
            }
            for name in names
        },
        "representative_single_proof_certificates": representative,
    }
    (out / "single_proof_v2_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    # Compact CSV for plotting/tables.
    with open(out / "path_metrics.csv", "w", encoding="utf-8") as f:
        f.write("path,median_prove_ms,median_inference_ms,median_verify_ms,proof_size_bytes,r1cs_constraints,test_accuracy,route_count\n")
        for idx, name in enumerate(names):
            test_acc = float((test_logits[name].argmax(axis=1) == test_labels).mean())
            f.write(
                f"{name},{costs[name]['median_prove_ms']},{costs[name]['median_inference_ms']},"
                f"{costs[name]['median_verify_ms']},{costs[name]['proof_size_bytes']},"
                f"{costs[name]['r1cs_constraints']},{test_acc},{int((chosen == idx).sum())}\n"
            )

    print(json.dumps(result["protocol_v2_single_proof"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
