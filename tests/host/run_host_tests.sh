#!/usr/bin/env bash
# Host (no-GPU) tests of the PCANI v2 gadgets, statement and MPC ceremony.
# Uses tests/host/shim/cuda_runtime.h in place of the CUDA toolkit, then runs the
# independent Python underconstraint checker on the exported circuits.
#   tests/host/run_host_tests.sh [out_dir]      (CXX defaults to g++)
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${1:-$ROOT/build-host}"
CXX="${CXX:-g++}"
mkdir -p "$OUT"
# trained real-ReLU model (roadmap 2.1) for the C++ <-> Python exactness check
python3 "$ROOT/python/export_trained_relu_model.py" --out "$OUT/relu_digits" > "$OUT/relu_digits_report.txt"
for t in gadgets statement_v2 ceremony; do
  "$CXX" -std=c++17 -O2 -Wall -Wno-unused-variable -I"$ROOT/tests/host/shim" -I"$ROOT/src" \
    "$ROOT/tests/host/test_$t.cpp" -o "$OUT/test_host_$t"
  "$OUT/test_host_$t" "$OUT" "$OUT/relu_digits" | grep -v "Constraint .* failed"
done
"$CXX" -std=c++17 -O2 -Wno-unused-variable -I"$ROOT/tests/host/shim" -I"$ROOT/src" -x c++ \
  "$ROOT/src/cli/ceremony.cu" -o "$OUT/zkml-ceremony-host"
python3 "$ROOT/tests/test_underconstrained.py" "$OUT"
echo "[ALL HOST TESTS PASS]"
