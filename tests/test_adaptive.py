import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

from zkml.adaptive import (
    PathSpec,
    calibrate_policy,
    evaluate_policy,
    estimate_r1cs_constraints,
)
from zkml.model import Model


def _logits(preds, confidences, n_classes=3):
    arr = np.zeros((len(preds), n_classes), dtype=np.float64)
    for i, (pred, conf) in enumerate(zip(preds, confidences)):
        conf = float(conf)
        # With all non-selected logits equal to zero, this logit gives the
        # selected class the requested softmax probability exactly.
        arr[i, pred] = np.log(conf * (n_classes - 1) / (1.0 - conf))
    return arr


def test_calibration_reduces_expected_proof_cost_without_exceeding_budget():
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    # Cheap path is very confident and correct on 7/10 samples, uncertain on 3.
    cheap_preds = np.array([0, 1, 2, 0, 1, 2, 0, 0, 0, 1])
    cheap_conf = np.array([.99, .99, .98, .97, .96, .95, .94, .40, .42, .38])
    mid_preds = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 1])
    mid_conf = np.array([.99, .99, .99, .98, .97, .96, .95, .90, .89, .45])
    full_preds = labels.copy()
    full_conf = np.full(10, .99)

    logits = {
        "cheap": _logits(cheap_preds, cheap_conf),
        "mid": _logits(mid_preds, mid_conf),
        "full": _logits(full_preds, full_conf),
    }
    paths = [
        PathSpec("cheap", 10.0),
        PathSpec("mid", 30.0),
        PathSpec("full", 100.0),
    ]
    policy = calibrate_policy(
        logits,
        labels,
        paths,
        max_accuracy_drop=0.0,
        threshold_grid=[0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
    )
    ev = evaluate_policy(logits, labels, paths, policy.thresholds)
    assert ev.accuracy == 1.0
    assert ev.expected_proof_cost < 100.0
    assert ev.proof_cost_savings > 0.0
    assert sum(ev.route_counts.values()) == len(labels)


def test_policy_digest_is_deterministic():
    labels = np.array([0, 1])
    logits = {
        "cheap": _logits([0, 0], [.99, .4], 2),
        "full": _logits([0, 1], [.99, .99], 2),
    }
    paths = [PathSpec("cheap", 1.0), PathSpec("full", 5.0)]
    a = calibrate_policy(logits, labels, paths, max_accuracy_drop=0.0, threshold_grid=[.5, .9])
    b = calibrate_policy(logits, labels, paths, max_accuracy_drop=0.0, threshold_grid=[.5, .9])
    assert a.digest() == b.digest()


def test_constraint_estimator_orders_reference_models():
    mnist = Model.create_mnist_mlp()
    cifar = Model.create_cifar_mlp()
    tiny = Model.create_tiny_transformer()
    assert estimate_r1cs_constraints(mnist) > 0
    assert estimate_r1cs_constraints(cifar) > estimate_r1cs_constraints(mnist)
    assert estimate_r1cs_constraints(tiny) > 0


def test_proof_chain_cost_is_cumulative_not_chosen_path_only():
    labels = np.array([0, 1])
    logits = {
        "cheap": np.array([[10.0, 0.0], [1.0, 0.0]]),
        "mid": np.array([[10.0, 0.0], [0.0, 10.0]]),
        "full": np.array([[10.0, 0.0], [0.0, 10.0]]),
    }
    paths = [PathSpec("cheap", 2.0), PathSpec("mid", 5.0), PathSpec("full", 11.0)]
    # sample 0 accepts cheap (margin 10); sample 1 rejects cheap and accepts mid.
    ev = evaluate_policy(logits, labels, paths, [5.0, 5.0], confidence_kind="margin")
    assert ev.route_counts == {"cheap": 1, "mid": 1, "full": 0}
    assert ev.expected_proof_cost == (2.0 + (2.0 + 5.0)) / 2.0


def test_selected_proof_cost_is_not_cumulative():
    labels = np.array([0, 1])
    logits = {
        "cheap": np.array([[10.0, 0.0], [1.0, 0.0]]),
        "mid": np.array([[10.0, 0.0], [0.0, 10.0]]),
        "full": np.array([[10.0, 0.0], [0.0, 10.0]]),
    }
    paths = [PathSpec("cheap", 2.0), PathSpec("mid", 5.0), PathSpec("full", 11.0)]
    ev = evaluate_policy(
        logits, labels, paths, [5.0, 5.0], confidence_kind="margin",
        cost_semantics="selected_proof",
    )
    assert ev.route_counts == {"cheap": 1, "mid": 1, "full": 0}
    assert ev.expected_proof_cost == (2.0 + 5.0) / 2.0
