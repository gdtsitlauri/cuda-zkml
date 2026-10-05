"""Proof-cost-aware adaptive inference research utilities.

This module implements the *policy layer* of Proof-Carrying Adaptive Neural
Inference (PCANI): among nested/candidate inference paths, route each sample to
the cheapest path whose confidence clears a calibrated threshold.  Thresholds
are selected using calibration data to minimize expected proving cost subject to
an accuracy-degradation budget relative to the full path.

This module is the policy/calibration layer only.  Cryptographic enforcement is
implemented separately. Protocol v1 uses a prefix proof chain; protocol v2 uses
a single selected-path proof whose public scores must satisfy that path's
calibrated acceptance predicate. Keeping policy search and proof verification
separate makes the evidence boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from itertools import product
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class PathSpec:
    name: str
    proof_cost: float
    workload_id: str = ""
    model_digest: str = ""

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("path name must be non-empty")
        if self.proof_cost <= 0:
            raise ValueError("proof_cost must be positive")


@dataclass
class AdaptivePolicy:
    paths: List[PathSpec]
    thresholds: List[float]
    max_accuracy_drop: float
    calibration_accuracy: float
    full_path_accuracy: float
    expected_proof_cost: float
    confidence_kind: str = "softmax"
    cost_semantics: str = "proof_chain"

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["policy_digest"] = self.digest()
        return payload

    def digest(self) -> str:
        canonical = json.dumps(
            {
                "paths": [asdict(p) for p in self.paths],
                "thresholds": self.thresholds,
                "max_accuracy_drop": self.max_accuracy_drop,
                "confidence_kind": self.confidence_kind,
                "cost_semantics": self.cost_semantics,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True)
            f.write("\n")


@dataclass
class AdaptiveEvaluation:
    accuracy: float
    full_path_accuracy: float
    accuracy_drop: float
    expected_proof_cost: float
    full_path_cost: float
    proof_cost_savings: float
    route_counts: Dict[str, int]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def softmax_confidence(logits: np.ndarray) -> np.ndarray:
    """Return max-class probability using a stable softmax."""
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2:
        raise ValueError("logits must have shape [n_samples, n_classes]")
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    probs = exp / exp.sum(axis=1, keepdims=True)
    return probs.max(axis=1)


def margin_confidence(logits: np.ndarray) -> np.ndarray:
    """Exact top-1 minus top-2 score margin.

    Unlike softmax confidence, this remains meaningful for large integer/field
    score magnitudes and is therefore the preferred routing statistic for the
    proof-chain protocol.
    """
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [n_samples, n_classes>=2]")
    part = np.partition(logits, kth=logits.shape[1] - 2, axis=1)
    return part[:, -1] - part[:, -2]


def confidence_values(logits: np.ndarray, kind: str = "softmax") -> np.ndarray:
    kind = str(kind).lower()
    if kind == "softmax":
        return softmax_confidence(logits)
    if kind == "margin":
        return margin_confidence(logits)
    raise ValueError(f"unsupported confidence kind: {kind}")


def _validate_logits(
    logits_by_path: Mapping[str, np.ndarray],
    paths: Sequence[PathSpec],
    labels: np.ndarray,
) -> Tuple[int, int]:
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    n = labels.shape[0]
    n_classes = None
    for p in paths:
        if p.name not in logits_by_path:
            raise ValueError(f"missing logits for path {p.name}")
        arr = np.asarray(logits_by_path[p.name])
        if arr.ndim != 2 or arr.shape[0] != n:
            raise ValueError(f"invalid logits shape for {p.name}: {arr.shape}")
        if n_classes is None:
            n_classes = arr.shape[1]
        elif arr.shape[1] != n_classes:
            raise ValueError("all paths must have the same number of classes")
    if not paths:
        raise ValueError("at least one path is required")
    return n, int(n_classes or 0)


def route_indices(
    logits_by_path: Mapping[str, np.ndarray],
    paths: Sequence[PathSpec],
    thresholds: Sequence[float],
    *,
    confidence_kind: str = "softmax",
) -> np.ndarray:
    """Route each sample from cheapest to most expensive path.

    The final path is a mandatory fallback and therefore has no threshold.
    """
    if len(thresholds) != max(0, len(paths) - 1):
        raise ValueError("threshold count must be number of paths minus one")
    n = next(iter(logits_by_path.values())).shape[0]
    chosen = np.full(n, len(paths) - 1, dtype=np.int64)
    unresolved = np.ones(n, dtype=bool)
    for idx, threshold in enumerate(thresholds):
        conf = confidence_values(np.asarray(logits_by_path[paths[idx].name]), confidence_kind)
        take = unresolved & (conf >= float(threshold))
        chosen[take] = idx
        unresolved[take] = False
    return chosen


def evaluate_policy(
    logits_by_path: Mapping[str, np.ndarray],
    labels: np.ndarray,
    paths: Sequence[PathSpec],
    thresholds: Sequence[float],
    *,
    confidence_kind: str = "softmax",
    cost_semantics: str = "proof_chain",
) -> AdaptiveEvaluation:
    labels = np.asarray(labels, dtype=np.int64)
    n, _ = _validate_logits(logits_by_path, paths, labels)
    chosen = route_indices(logits_by_path, paths, thresholds, confidence_kind=confidence_kind)

    preds = np.empty(n, dtype=np.int64)
    costs = np.empty(n, dtype=np.float64)
    route_counts: Dict[str, int] = {}
    for idx, p in enumerate(paths):
        mask = chosen == idx
        route_counts[p.name] = int(mask.sum())
        if mask.any():
            preds[mask] = np.asarray(logits_by_path[p.name])[mask].argmax(axis=1)
            if cost_semantics == "proof_chain":
                # Protocol v1: reaching path i carries every attempted prefix proof.
                costs[mask] = sum(float(q.proof_cost) for q in paths[: idx + 1])
            elif cost_semantics == "selected_proof":
                # Protocol v2: only the selected model proof is transmitted and
                # verified. The verifier checks that proof-bound public scores
                # satisfy the selected path's calibrated acceptance threshold.
                costs[mask] = float(p.proof_cost)
            else:
                raise ValueError(f"unsupported cost semantics: {cost_semantics}")

    full_logits = np.asarray(logits_by_path[paths[-1].name])
    full_preds = full_logits.argmax(axis=1)
    acc = float((preds == labels).mean())
    full_acc = float((full_preds == labels).mean())
    expected_cost = float(costs.mean())
    full_cost = float(paths[-1].proof_cost)
    return AdaptiveEvaluation(
        accuracy=acc,
        full_path_accuracy=full_acc,
        accuracy_drop=full_acc - acc,
        expected_proof_cost=expected_cost,
        full_path_cost=full_cost,
        proof_cost_savings=1.0 - expected_cost / full_cost,
        route_counts=route_counts,
    )


def calibrate_policy(
    logits_by_path: Mapping[str, np.ndarray],
    labels: np.ndarray,
    paths: Sequence[PathSpec],
    *,
    max_accuracy_drop: float = 0.005,
    threshold_grid: Iterable[float] | Mapping[str, Iterable[float]] | None = None,
    confidence_kind: str = "softmax",
    cost_semantics: str = "proof_chain",
) -> AdaptivePolicy:
    """Calibrate routing thresholds by constrained proof-cost minimization.

    Objective:
        minimize E[C_ZK(route(x))]
        subject to Acc(policy) >= Acc(full) - max_accuracy_drop.

    The search is exact over the supplied threshold grid and deterministic.
    Candidate paths must be ordered by increasing proof cost; the final path is
    the full/reference model and is always used as fallback.
    """
    labels = np.asarray(labels, dtype=np.int64)
    _validate_logits(logits_by_path, paths, labels)
    paths = list(paths)
    if any(paths[i].proof_cost > paths[i + 1].proof_cost for i in range(len(paths) - 1)):
        raise ValueError("paths must be ordered by non-decreasing proof cost")
    if max_accuracy_drop < 0:
        raise ValueError("max_accuracy_drop must be non-negative")

    if len(paths) == 1:
        ev = evaluate_policy(logits_by_path, labels, paths, [], confidence_kind=confidence_kind, cost_semantics=cost_semantics)
        return AdaptivePolicy(paths, [], max_accuracy_drop, ev.accuracy,
                              ev.full_path_accuracy, ev.expected_proof_cost, confidence_kind, cost_semantics)

    if threshold_grid is None:
        if confidence_kind == "softmax":
            per_path_grids = [
                [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85,
                 0.90, 0.925, 0.95, 0.975, 0.99, 1.0000001]
                for _ in paths[:-1]
            ]
        elif confidence_kind == "margin":
            # Margins are path-scale dependent.  Build a separate deterministic
            # quantile grid for each non-final path rather than forcing a single
            # arbitrary numeric scale across architectures.
            quantiles = np.array([0.0, .05, .10, .20, .30, .40, .50, .60, .70, .80, .90, .95, .99, 1.0])
            per_path_grids = []
            for p in paths[:-1]:
                conf = confidence_values(np.asarray(logits_by_path[p.name]), "margin")
                vals = np.unique(np.quantile(conf, quantiles)).astype(float).tolist()
                vals.append(float(np.max(conf)) + 1.0)  # deterministic reject-all sentinel
                per_path_grids.append(vals)
        else:
            raise ValueError(f"unsupported confidence kind: {confidence_kind}")
    elif isinstance(threshold_grid, Mapping):
        per_path_grids = []
        for p in paths[:-1]:
            if p.name not in threshold_grid:
                raise ValueError(f"missing threshold grid for path {p.name}")
            vals = [float(v) for v in threshold_grid[p.name]]
            if not vals:
                raise ValueError(f"empty threshold grid for path {p.name}")
            per_path_grids.append(vals)
    else:
        grid = [float(v) for v in threshold_grid]
        if not grid:
            raise ValueError("threshold grid must be non-empty")
        if confidence_kind == "softmax" and any((t < 0 or t > 1) for t in grid):
            raise ValueError("softmax thresholds must lie in [0, 1]")
        per_path_grids = [grid for _ in paths[:-1]]

    full_preds = np.asarray(logits_by_path[paths[-1].name]).argmax(axis=1)
    full_acc = float((full_preds == labels).mean())
    floor = full_acc - max_accuracy_drop

    best = None
    for thresholds in product(*per_path_grids):
        # For softmax probabilities a monotonic threshold schedule is easy to
        # interpret. Raw margins have architecture-dependent scales, so no such
        # cross-path ordering is imposed.
        if confidence_kind == "softmax" and any(
            thresholds[i] > thresholds[i + 1] for i in range(len(thresholds) - 1)
        ):
            continue
        ev = evaluate_policy(logits_by_path, labels, paths, thresholds, confidence_kind=confidence_kind, cost_semantics=cost_semantics)
        if ev.accuracy + 1e-12 < floor:
            continue
        candidate = (ev.expected_proof_cost, -ev.accuracy, tuple(thresholds), ev)
        if best is None or candidate[:3] < best[:3]:
            best = candidate

    if best is None:
        # Guaranteed-safe fallback: route everything to the full path.
        if confidence_kind == "softmax":
            thresholds = [1.0000001] * (len(paths) - 1)
        else:
            thresholds = [
                float(np.max(confidence_values(np.asarray(logits_by_path[p.name]), confidence_kind))) + 1.0
                for p in paths[:-1]
            ]
        ev = evaluate_policy(logits_by_path, labels, paths, thresholds, confidence_kind=confidence_kind, cost_semantics=cost_semantics)
    else:
        thresholds = list(best[2])
        ev = best[3]

    return AdaptivePolicy(
        paths=paths,
        thresholds=list(thresholds),
        max_accuracy_drop=float(max_accuracy_drop),
        calibration_accuracy=ev.accuracy,
        full_path_accuracy=ev.full_path_accuracy,
        expected_proof_cost=ev.expected_proof_cost,
        confidence_kind=confidence_kind,
        cost_semantics=cost_semantics,
    )


def estimate_r1cs_constraints(model) -> int:
    """Estimate constraints using the repository's current CircuitBuilder rules.

    Exact for linear/ReLU/softmax layers as currently implemented.  The
    self-attention estimate mirrors the projection/score/softmax/value/output
    structure of CircuitBuilder and is intended as a cost proxy until measured
    prover-time calibration is available.
    """
    total = 0
    for layer in model.layers:
        kind = layer.layer_type
        if kind == "linear":
            total += layer.out_features * layer.in_features  # products
            total += layer.out_features  # output linear constraints
        elif kind == "relu":
            total += 3 * layer.in_features  # x^2, x^3, polynomial equality
        elif kind == "softmax":
            n = layer.in_features
            total += 4 * n + 2
        elif kind == "self_attention":
            seq = int(layer.config["seq_len"])
            hidden = int(layer.config["hidden_size"])
            heads = int(layer.config.get("num_heads", 1))
            if heads <= 0 or hidden % heads:
                raise ValueError("invalid self-attention head configuration")
            head_dim = hidden // heads
            projections_qkv = 3 * seq * hidden * (hidden + 1)
            scores = seq * heads * seq * (head_dim + 1)
            row_softmax = seq * heads * (4 * seq + 2)
            weighted_values = seq * heads * head_dim * (seq + 1)
            output_projection = seq * hidden * (hidden + 1)
            total += projections_qkv + scores + row_softmax + weighted_values + output_projection
        else:
            raise ValueError(f"unsupported layer type: {kind}")
    return int(total)
