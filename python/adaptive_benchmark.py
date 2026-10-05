#!/usr/bin/env python3
"""Calibrate/evaluate the PCANI proof-cost-aware routing policy.

Examples
--------
Synthetic sanity check (NOT a research benchmark):
    python adaptive_benchmark.py --demo --output ../benchmarks/results/adaptive_demo.json

Real logits:
    python adaptive_benchmark.py \
        --calibration calibration.npz --test test.npz \
        --paths paths.json --max-accuracy-drop 0.005 \
        --policy-out policy.json --output results.json

NPZ files must contain `labels` and one `<path_name>_logits` array per path.
`paths.json` is a list of {name, proof_cost, workload_id?, model_digest?} objects.
Use *measured prover time* as proof_cost when available; structural constraint
counts are acceptable only as a clearly labelled proxy during development.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from zkml.adaptive import PathSpec, calibrate_policy, evaluate_policy


def load_paths(path: str) -> List[PathSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("paths JSON must be a list")
    return [PathSpec(**item) for item in payload]


def load_npz(path: str, paths: List[PathSpec]) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    with np.load(path) as data:
        if "labels" not in data:
            raise ValueError(f"{path} is missing labels")
        labels = np.asarray(data["labels"], dtype=np.int64)
        logits = {}
        for p in paths:
            key = f"{p.name}_logits"
            if key not in data:
                raise ValueError(f"{path} is missing {key}")
            logits[p.name] = np.asarray(data[key], dtype=np.float64)
    return labels, logits


def _demo_logits(seed: int = 2026):
    rng = np.random.default_rng(seed)
    n, classes = 600, 5
    labels = rng.integers(0, classes, size=n)
    difficulty = rng.uniform(0, 1, size=n)

    def make_path(error_scale: float, confidence_bias: float):
        logits = rng.normal(0, 0.35, size=(n, classes))
        correct_prob = np.clip(1.0 - error_scale * difficulty, 0.05, 0.995)
        chosen = labels.copy()
        errors = rng.random(n) > correct_prob
        chosen[errors] = (labels[errors] + rng.integers(1, classes, size=errors.sum())) % classes
        margin = confidence_bias + 5.0 * (1.0 - difficulty)
        logits[np.arange(n), chosen] += margin
        return logits

    # Cheap path degrades on hard samples; the full path is much more stable.
    logits = {
        "early": make_path(0.42, 0.5),
        "mid": make_path(0.18, 1.2),
        "full": make_path(0.04, 2.0),
    }
    paths = [
        PathSpec("early", 0.30, workload_id="synthetic-demo"),
        PathSpec("mid", 0.62, workload_id="synthetic-demo"),
        PathSpec("full", 1.00, workload_id="synthetic-demo"),
    ]
    split = n // 2
    cal = labels[:split], {k: v[:split] for k, v in logits.items()}
    test = labels[split:], {k: v[split:] for k, v in logits.items()}
    return paths, cal, test


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--calibration")
    ap.add_argument("--test")
    ap.add_argument("--paths")
    ap.add_argument("--max-accuracy-drop", type=float, default=0.005)
    ap.add_argument("--confidence-kind", choices=["softmax", "margin"], default="softmax")
    ap.add_argument("--policy-out")
    ap.add_argument("--output")
    args = ap.parse_args()

    if args.demo:
        paths, (cal_y, cal_logits), (test_y, test_logits) = _demo_logits()
        evidence = "synthetic_sanity_check_only"
    else:
        if not (args.calibration and args.test and args.paths):
            ap.error("real evaluation requires --calibration, --test and --paths")
        paths = load_paths(args.paths)
        cal_y, cal_logits = load_npz(args.calibration, paths)
        test_y, test_logits = load_npz(args.test, paths)
        evidence = "user_supplied_logits"

    policy = calibrate_policy(
        cal_logits,
        cal_y,
        paths,
        max_accuracy_drop=args.max_accuracy_drop,
        confidence_kind=args.confidence_kind,
    )
    test_eval = evaluate_policy(
        test_logits, test_y, paths, policy.thresholds,
        confidence_kind=args.confidence_kind,
    )
    report = {
        "algorithm": "PCANI proof-cost-aware routing policy",
        "status": "policy-calibration-layer",
        "evidence": evidence,
        "security_note": (
            "This command calibrates/evaluates the routing policy only. "
            "Cryptographic route enforcement is provided by PCANI protocol v1 "
            "(python/pcani_verify.py) using model-specific Groth16 proofs linked "
            "by the same public quantized input."
        ),
        "policy": policy.to_dict(),
        "test": test_eval.to_dict(),
    }

    if args.policy_out:
        Path(args.policy_out).parent.mkdir(parents=True, exist_ok=True)
        policy.save(args.policy_out)
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
