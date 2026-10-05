#!/usr/bin/env python3
"""Fail-closed benchmark comparator.

Only emits speedup ratios for rows with the same non-empty workload_id and
matching semantic metadata. Historical repository rows intentionally lack these
fields and are therefore reported as non-comparable.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

REQUIRED_MATCH = [
    "workload_id", "precision", "statement_semantics", "batch_size",
    "setup_accounting"
]


def comparable(a, b):
    reasons = []
    for key in REQUIRED_MATCH:
        av, bv = a.get(key), b.get(key)
        if av in (None, "") or bv in (None, ""):
            reasons.append(f"missing {key}")
        elif av != bv:
            reasons.append(f"different {key}: {av!r} vs {bv!r}")
    return not reasons, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json")
    ap.add_argument("--a", required=True, help="exact system name A")
    ap.add_argument("--b", required=True, help="exact system name B")
    args = ap.parse_args()
    data = json.loads(Path(args.json).read_text(encoding="utf-8"))
    rows = data.get("results", data if isinstance(data, list) else [])
    idx = {r.get("system"): r for r in rows}
    if args.a not in idx or args.b not in idx:
        raise SystemExit("requested system row not found")
    a, b = idx[args.a], idx[args.b]
    ok, reasons = comparable(a, b)
    out = {"comparable": ok, "reasons": reasons, "a": args.a, "b": args.b}
    if ok:
        for metric in ("avg_prove_ms", "avg_verify_ms", "proof_size_bytes"):
            av, bv = a.get(metric), b.get(metric)
            if isinstance(av, (int, float)) and isinstance(bv, (int, float)) and av > 0 and bv > 0:
                out[f"{metric}_a_over_b"] = av / bv
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0 if ok else 2

if __name__ == "__main__":
    raise SystemExit(main())
