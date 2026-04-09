#!/usr/bin/env python3
"""
Export CUDA-zkML proof artifacts into Solidity/Ethers-friendly JSON files.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zkml.artifacts import export_solidity_bundle


def main() -> int:
    parser = argparse.ArgumentParser(description="Export zkML artifacts for Solidity verification")
    parser.add_argument("--vk", required=True, help="Path to vk.bin")
    parser.add_argument("--proof", required=True, help="Path to proof.bin")
    parser.add_argument("--public-inputs", required=True, help="Path to public_inputs.bin or .json")
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where Solidity JSON artifacts will be written",
    )
    args = parser.parse_args()

    bundle = export_solidity_bundle(
        vk_path=args.vk,
        proof_path=args.proof,
        public_inputs_path=args.public_inputs,
        output_dir=args.output_dir,
    )

    print("Solidity artifacts exported successfully.")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Public inputs: {len(bundle['publicInputs'])}")
    print(f"  IC points: {len(bundle['vk']['ic'])}")
    print("  Files:")
    print("    vk.solidity.json")
    print("    proof.solidity.json")
    print("    public_inputs.solidity.json")
    print("    calldata.solidity.json")
    print("    verifier_call.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
