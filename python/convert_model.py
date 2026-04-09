#!/usr/bin/env python3
"""Convert supported model formats to CUDA-zkML flat float32 binary weights."""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from zkml.model import Model


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert model to CUDA-zkML binary format")
    parser.add_argument("--input", required=True, help="Input model path (.onnx or .bin)")
    parser.add_argument("--output", required=True, help="Output .bin path")
    parser.add_argument(
        "--arch-output",
        default=None,
        help="Optional architecture sidecar output path (defaults to OUTPUT.arch.json)",
    )
    args = parser.parse_args()

    in_path = args.input
    out_path = args.output
    arch_out_path = args.arch_output or (out_path + ".arch.json")
    ext = os.path.splitext(in_path)[1].lower()

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(arch_out_path) or ".", exist_ok=True)

    if ext == ".bin":
        shutil.copyfile(in_path, out_path)
        arch_in_path = in_path + ".arch.json"
        if os.path.exists(arch_in_path):
            shutil.copyfile(arch_in_path, arch_out_path)
            print(f"Copied architecture sidecar: {arch_out_path}")
        print(f"Copied binary model: {out_path}")
        return 0

    if ext == ".onnx":
        model = Model.from_onnx(in_path)
        model.to_binary(out_path)
        model.save_architecture(arch_out_path)
        print(f"Converted ONNX model to binary: {out_path}")
        print(f"Saved architecture sidecar: {arch_out_path}")
        return 0

    print(f"Unsupported model format: {in_path}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
