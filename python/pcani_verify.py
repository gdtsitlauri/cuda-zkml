#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from zkml.pcani_protocol import verify_route_bundle


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify a PCANI proof-chain route bundle")
    ap.add_argument("--bundle", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--verify-cmd", default="./build/zkml-verify")
    ap.add_argument("--report", default="")
    args = ap.parse_args()

    report = verify_route_bundle(args.bundle, args.manifest, args.verify_cmd)
    payload = report.to_dict()
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
    return 0 if report.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
