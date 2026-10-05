#!/usr/bin/env bash
# Roadmap phases 1-2 validation in one command, on any NVIDIA GPU (fail-closed).
#
#   1. clean Release build for the detected GPU (zkml-prove, zkml-verify, zkml-ceremony)
#   2. ctest (incl. host tests of gadgets / statement v2 / MPC ceremony) + pytest
#   3. host tests without CUDA + independent Python underconstraint check
#   4. Solidity: compile Verifier.sol and PCANIContextVerifier.sol (solc 0.8.x)
#   5. end-to-end GPU experiments E1-E6 (python/run_phase12_e2e.py)
#   6. archive Phase12_<gpu>_<date>.zip
#
#   CUDA_ARCH=75 overrides the compute-capability auto-detection.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
command -v nvidia-smi >/dev/null
command -v nvcc >/dev/null
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_' | sed 's/_*$//')"
if [[ -z "${CUDA_ARCH:-}" ]]; then
  CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.[:space:]')"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/phase12/${GPU}_${STAMP}"
mkdir -p "$OUT"
{ echo "gpu=$GPU"; echo "cuda_arch=$CUDA_ARCH"; nvidia-smi; nvcc --version; } > "$OUT/environment.txt" 2>&1

python -m pip install -q -r requirements.txt py-ecc scikit-learn pytest py-solc-x
command -v zip >/dev/null || { apt-get -qq install -y zip >/dev/null 2>&1 || true; }

echo "== 1-2. build + ctest + pytest"
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DZKML_FAST_DEBUG_LOOP=OFF -DZKML_FAST_ITERATION=OFF -DZKML_ULTRA_FAST_COMPILE=OFF \
  -DZKML_NVCC_THREADS=1 -DZKML_NVCC_SPLIT_COMPILE_THREADS=1
cmake --build build -j2
ctest --test-dir build --output-on-failure | tee "$OUT/ctest.log"

echo "== 3. host tests (no CUDA) + underconstraint check"
CXX=g++ bash tests/host/run_host_tests.sh "$OUT/host" | tee "$OUT/host_tests.log"
PCANI_R1CS_DIR="$OUT/host" PYTHONPATH=python python -m pytest -q | tee "$OUT/pytest.log"

echo "== 4. Solidity compile"
python - <<'PY' | tee "$OUT/solidity.log"
import solcx
solcx.install_solc("0.8.24")
out = solcx.compile_files(["contracts/Verifier.sol", "contracts/PCANIContextVerifier.sol"],
                          output_values=["abi", "bin"], solc_version="0.8.24", allow_paths=["contracts"])
for k in out:
    print(k, "bytecode", len(out[k]["bin"]) // 2, "bytes")
print("[SOLIDITY COMPILE PASS]")
PY

echo "== 5. end-to-end GPU experiments"
PYTHONPATH=python python python/run_phase12_e2e.py --build-dir build --out "$OUT/e2e" 2>&1 | tee "$OUT/e2e.log"
grep -q "PHASE12 E2E PASS" "$OUT/e2e.log"

ZIP="Phase12_${GPU}_${STAMP}.zip"
zip -qr "$ZIP" "$OUT" -x '*.bin' '*.r1cs' '*.wtns'
echo "[PHASE12 DONE] $ZIP"
