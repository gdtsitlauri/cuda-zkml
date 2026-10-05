#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

command -v nvidia-smi >/dev/null
command -v nvcc >/dev/null
nvidia-smi
nvcc --version

python -m pip install -q numpy scikit-learn pytest
rm -rf build
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=75 \
  -DZKML_FAST_DEBUG_LOOP=OFF \
  -DZKML_FAST_ITERATION=OFF \
  -DZKML_ULTRA_FAST_COMPILE=OFF \
  -DZKML_NVCC_THREADS=1 \
  -DZKML_NVCC_SPLIT_COMPILE_THREADS=1
cmake --build build -j2
PYTHONPATH=python python -m pytest -q
ctest --test-dir build --output-on-failure
PYTHONPATH=python python python/pcani_colab_experiment.py \
  --build-dir build \
  --benchmark-dir benchmarks/pcani_digits \
  --results-dir results/pcani_colab_final \
  --widths 16,48,96,160 \
  --repeats 5 \
  --max-accuracy-drop 0.0 \
  --heldout-accuracy-drop-limit 0.01 \
  --max-route-bundles 4

echo "Result: $ROOT/PCANI_final_results.zip"
