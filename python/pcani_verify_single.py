#!/usr/bin/env python3
import argparse, json
from zkml.pcani_protocol_v2 import verify_single_proof_certificate

ap = argparse.ArgumentParser(description="Verify a PCANI v2 single-proof route certificate")
ap.add_argument("--certificate", required=True)
ap.add_argument("--manifest", required=True)
ap.add_argument("--verify-cmd", required=True)
args = ap.parse_args()
report = verify_single_proof_certificate(args.certificate, args.manifest, args.verify_cmd)
print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
raise SystemExit(0 if report.valid else 1)
