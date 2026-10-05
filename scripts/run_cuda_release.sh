#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
rm -rf build-release
cmake -S . -B build-release -DCMAKE_BUILD_TYPE=Release \
  -DZKML_FAST_DEBUG_LOOP=OFF -DZKML_FAST_ITERATION=OFF -DZKML_ULTRA_FAST_COMPILE=OFF
cmake --build build-release -j"$(nproc)"
ctest --test-dir build-release --output-on-failure
pytest -q tests
mkdir -p results
{
  echo "commit_or_tree: research-release"
  echo "date_utc: $(date -u +%FT%TZ)"
  nvcc --version || true
  nvidia-smi || true
} > results/cuda_environment.txt
# Run a minimal proof smoke test if binaries exist.
./build-release/zkml-prove --demo | tee results/cuda_demo.log
