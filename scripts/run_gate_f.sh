#!/usr/bin/env bash
# Gate F (publication-strength replication) in one command, on any NVIDIA GPU.
#
#   1. clean Release build for the detected GPU architecture + pytest + ctest
#   2. PCANI protocol-v2 runs (fail-closed, exact CUDA/Python logits):
#        digits      the completed T4 Digits experiment, unchanged  -> GPU replication
#        mnist_mlp   MNIST 784-input random-feature MLP paths 16/48/96/160
#        mnist_conv  MNIST lowered-convolution paths 2/4/8/12 filters (hidden 128..768)
#   3. matched EZKL comparison on the same exact-integer models (CPU prover)
#   4. one archive: GateF_<gpu>_<date>.zip
#
# Environment overrides:
#   WORKLOADS="digits mnist_mlp mnist_conv"   subset to run
#   REPEATS=5            timing repeats per path
#   EZKL=1               0 skips the EZKL comparison
#   EZKL_PATHS=""        e.g. "p16,p160" to limit EZKL to some paths
#   CUDA_ARCH=75         override auto-detection (nvidia-smi compute_cap)
#   MAX_TEST=5000        MNIST test samples used for calibration/evaluation
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
WORKLOADS="${WORKLOADS:-digits mnist_mlp mnist_conv}"
REPEATS="${REPEATS:-5}"
EZKL="${EZKL:-1}"
EZKL_PATHS="${EZKL_PATHS:-}"
MAX_TEST="${MAX_TEST:-5000}"

command -v nvidia-smi >/dev/null
command -v nvcc >/dev/null
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr -c 'A-Za-z0-9\n' '_' | sed 's/_*$//')"
if [[ -z "${CUDA_ARCH:-}" ]]; then
  CUDA_ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '.[:space:]')"
fi
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
OUT="results/gate_f/${GPU}_${STAMP}"
mkdir -p "$OUT"
{
  echo "gpu=$GPU"; echo "cuda_arch=$CUDA_ARCH"; echo "workloads=$WORKLOADS"; echo "repeats=$REPEATS"
  nvidia-smi; nvcc --version; git rev-parse HEAD 2>/dev/null || true
} > "$OUT/environment.txt" 2>&1

# Same dependency set as PCANI_Colab_Final_Validation.ipynb (the full pytest
# suite imports web3/solc/py-ecc), plus scikit-learn for the benchmarks.
python -m pip install -q -r requirements.txt py-ecc scikit-learn pytest onnx
if [[ "$EZKL" == "1" ]]; then python -m pip install -q "ezkl>=23" || EZKL=0; fi
command -v zip >/dev/null || { apt-get -qq install -y zip >/dev/null 2>&1 || true; }

# 1. Build and regression
rm -rf build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
  -DZKML_FAST_DEBUG_LOOP=OFF -DZKML_FAST_ITERATION=OFF -DZKML_ULTRA_FAST_COMPILE=OFF \
  -DZKML_NVCC_THREADS=1 -DZKML_NVCC_SPLIT_COMPILE_THREADS=1
cmake --build build -j2
PYTHONPATH=python python -m pytest -q | tee "$OUT/pytest.log"
ctest --test-dir build --output-on-failure | tee "$OUT/ctest.log"

run_workload() {
  local name="$1" widths="$2" script="$3" prep_args="$4"
  echo "=== PCANI v2: $name ==="
  PYTHONPATH=python python python/pcani_colab_experiment.py \
    --build-dir build \
    --benchmark-dir "benchmarks/gate_f/$name" \
    --results-dir "$OUT/$name" \
    --widths "$widths" \
    --repeats "$REPEATS" \
    --max-accuracy-drop 0.0 \
    --heldout-accuracy-drop-limit 0.01 \
    --max-route-bundles 4 \
    --prepare-script "$script" \
    --prepare-args "$prep_args" \
    --archive-name "$OUT/${name}_pcani" 2>&1 | tee "$OUT/${name}_run.log"
  if [[ "$EZKL" == "1" ]]; then
    echo "=== matched EZKL: $name ==="
    PYTHONPATH=python python python/run_ezkl_matched.py \
      --benchmark-dir "benchmarks/gate_f/$name" \
      --output "$OUT/$name/ezkl_matched.json" \
      --repeats 3 --paths "$EZKL_PATHS" \
      --workdir ".zkml_tmp/ezkl_$name" 2>&1 | tee "$OUT/${name}_ezkl.log" || true
  fi
}

for w in $WORKLOADS; do
  case "$w" in
    digits)     run_workload digits "16,48,96,160" pcani_digits_prepare.py "" ;;
    mnist_mlp)  run_workload mnist_mlp "16,48,96,160" pcani_prepare.py \
                  "--dataset mnist --family mlp --max-test $MAX_TEST --data-home .cache/openml" ;;
    mnist_conv) run_workload mnist_conv "2,4,8,12" pcani_prepare.py \
                  "--dataset mnist --family conv --kernel 5 --stride 3 --max-test $MAX_TEST --data-home .cache/openml" ;;
    *) echo "unknown workload $w"; exit 2 ;;
  esac
done

python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
out = Path(sys.argv[1])
rows = []
for d in sorted(p for p in out.iterdir() if p.is_dir()):
    s = d / "run_summary.json"
    if not s.exists():
        continue
    r = json.loads(s.read_text())
    t = r["pcani"]["test"]
    e = d / "ezkl_matched.json"
    ez = json.loads(e.read_text()) if e.exists() else {}
    rows.append({
        "workload": d.name,
        "workload_id": r["benchmark"].get("workload_id", "digits-exact-integer"),
        "hypothesis_supported": r["hypothesis_supported_on_this_run"],
        "adaptive_accuracy": t["accuracy"], "full_path_accuracy": t["full_path_accuracy"],
        "accuracy_drop": t["accuracy_drop"], "proof_cost_savings": t["proof_cost_savings"],
        "median_prove_ms": {k: v["median_prove_ms"] for k, v in r["proof_costs"].items()},
        "ezkl_median_prove_ms": {row["path"]: row.get("median_prove_ms") for row in ez.get("rows", [])},
        "ezkl_comparable": {row["path"]: row.get("comparable") for row in ez.get("rows", [])},
    })
(out / "gate_f_summary.json").write_text(json.dumps(rows, indent=2) + "\n")
print(json.dumps(rows, indent=2))
PY

(cd "$(dirname "$OUT")" && zip -qr "$ROOT/GateF_$(basename "$OUT").zip" "$(basename "$OUT")")
echo "Gate F archive: $ROOT/GateF_$(basename "$OUT").zip"
