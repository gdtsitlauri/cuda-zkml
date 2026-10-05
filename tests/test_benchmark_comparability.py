import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
from compare_benchmarks import comparable


def test_unmatched_rows_fail_closed():
    ok, reasons = comparable({"workload_id": "a"}, {"workload_id": "b"})
    assert not ok
    assert reasons


def test_matched_semantics_are_comparable():
    base = {
        "workload_id": "w",
        "precision": "int8",
        "statement_semantics": "private-input-fixed-model",
        "batch_size": 1,
        "setup_accounting": "excluded",
    }
    ok, reasons = comparable(dict(base), dict(base))
    assert ok
    assert reasons == []
