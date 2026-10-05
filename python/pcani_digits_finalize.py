#!/usr/bin/env python3
"""Calibrate/evaluate PCANI using measured standalone prover costs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from zkml.adaptive import PathSpec, calibrate_policy, evaluate_policy


def load_logits(path: Path, names):
    with np.load(path) as data:
        y = np.asarray(data["labels"], dtype=np.int64)
        logits = {n: np.asarray(data[f"{n}_logits"], dtype=np.float64) for n in names}
    return y, logits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark-dir", default="benchmarks/pcani_digits")
    ap.add_argument("--proof-costs", required=True, help="JSON mapping path -> median prover milliseconds")
    ap.add_argument("--max-accuracy-drop", type=float, default=0.01)
    ap.add_argument(
        "--cost-semantics", choices=["proof_chain", "selected_proof"],
        default="selected_proof",
        help="Protocol-v1 cumulative prefix cost or protocol-v2 selected-proof cost",
    )
    args = ap.parse_args()

    root = Path(args.benchmark_dir)
    prep = json.loads((root / "prepare_report.json").read_text())
    costs = json.loads(Path(args.proof_costs).read_text())
    names = [p["name"] for p in prep["paths"]]
    # Reports from pcani_prepare.py carry their own identity; the original
    # Digits report predates these fields.
    benchmark = prep.get("benchmark", "sklearn-digits exact-integer PCANI")
    workload_id = prep.get("workload_id", "digits-exact-integer")
    paths = []
    for p in prep["paths"]:
        name = p["name"]
        raw = costs[name]
        cost = float(raw["median_prove_ms"] if isinstance(raw, dict) else raw)
        paths.append(PathSpec(name, cost, workload_id=workload_id, model_digest=p["model_sha256"]))

    cal_y, cal_logits = load_logits(root / "calibration.npz", names)
    test_y, test_logits = load_logits(root / "test.npz", names)
    policy = calibrate_policy(
        cal_logits, cal_y, paths,
        max_accuracy_drop=args.max_accuracy_drop,
        confidence_kind="margin",
        cost_semantics=args.cost_semantics,
    )
    test_eval = evaluate_policy(
        test_logits, test_y, paths, policy.thresholds,
        confidence_kind="margin",
        cost_semantics=args.cost_semantics,
    )

    result = {
        "benchmark": benchmark,
        "workload_id": workload_id,
        "evidence": "real_dataset_and_measured_gpu_proof_costs",
        "cost_semantics": (
            "PathSpec.proof_cost is standalone prover time. Adaptive route cost is cumulative "
            "over every attempted proof in the verified proof chain."
            if args.cost_semantics == "proof_chain" else
            "PathSpec.proof_cost is standalone prover time. Adaptive route cost is the selected "
            "model-specific proof only; route validity is checked by protocol-v2 acceptance rules."
        ),
        "policy": policy.to_dict(),
        "test": test_eval.to_dict(),
        "standalone_path_accuracy": {p["name"]: p["test_accuracy"] for p in prep["paths"]},
        "measured_proof_cost_ms": {p.name: p.proof_cost for p in paths},
    }
    (root / "policy.json").write_text(json.dumps(policy.to_dict(), indent=2, sort_keys=True) + "\n")
    (root / "final_results.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
