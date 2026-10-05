# Gate F runbook — publication-strength replication

Gate F items from `EXPERIMENT_PLAN_2026.md` and what now exists for each:

| Gate F item | Status |
|---|---|
| Repeat protocol v2 on another NVIDIA GPU | **Run only.** `scripts/run_gate_f.sh` auto-detects the GPU architecture; the `digits` workload re-runs the T4 experiment unchanged. |
| Larger benchmark / model family | **Run only.** `python/pcani_prepare.py`: MNIST with the Digits model family (`mnist_mlp`) and a new convolutional family lowered to the exact LINEAR circuit (`mnist_conv`). No CUDA change was needed. |
| Matched contemporary zkML comparison | **Run only.** `python/run_ezkl_matched.py` proves the *same* exact-integer models in EZKL with public input / fixed parameters / scale 0 and checks EZKL's outputs equal the exact logits (`comparable=true`). |
| Systematic literature/patent novelty audit | Reading task (see `NOVELTY_GATE_2026.md`); not code. |
| External cryptographic review | Human reviewer. |

## One command

Google Colab: open `PCANI_Colab_GateF.ipynb`, choose a GPU **other than T4**
(L4/A100), run all cells, keep the downloaded `GateF_<gpu>_<date>.zip`.

Any Linux machine with an NVIDIA GPU, CUDA toolkit, CMake and Python:

```bash
bash scripts/run_gate_f.sh
# subsets / overrides:
WORKLOADS="digits mnist_mlp" REPEATS=5 EZKL=1 bash scripts/run_gate_f.sh
```

Per workload the archive contains the usual fail-closed PCANI v2 outputs
(`run_summary.json`, measured per-path prover costs, policy, verified single-proof
certificates, logs) plus `ezkl_matched.json`, and a top-level
`gate_f_summary.json`.

## What was executed on 2026-10-03 (CPU only, no CUDA build)

* `tests/test_pcani_prepare.py`: the generalised generator reproduces the
  original T4 Digits models **byte-for-byte** and their logits exactly; the conv
  lowering equals a direct convolution; an independent pure-Python-integer
  evaluation of every exported model equals the stored logits; the finalizer
  accepts the new reports. `tests/test_ezkl_matched_export.py`: ONNX export
  reproduces the integer model. (The four pre-existing `test_pcani_protocol.py`
  tests use a POSIX shell stub and only run on Linux/Colab.)
* MNIST benchmark generation (20k train / 5k calibration / 5k official test):

  | Workload | Paths | Hidden width | Test accuracy |
  |---|---|---|---|
  | mnist_mlp | p16 / p48 / p96 / p160 | 16 / 48 / 96 / 160 | 60.8% / 77.1% / 85.8% / 88.7% |
  | mnist_conv (k=5, s=3) | p2 / p4 / p8 / p12 | 128 / 256 / 512 / 768 | 87.4% / 91.4% / 92.5% / 94.0% |

* Matched EZKL 23.0.5 (Windows CPU, 2 repeats), outputs equal exact logits on
  every row — `results/ezkl_matched_preliminary_2026-10-03/`:

  | Workload/path | logrows | median prove | verify | proof size |
  |---|---|---|---|---|
  | digits p16 | 15 | 1.30 s | 22 ms | 27.7 kB |
  | digits p160 | 15 | 1.36 s | 21 ms | 27.7 kB |
  | mnist_conv p2 | 16 | 2.89 s | 49 ms | 128 kB |
  | mnist_conv p12 | 19 | 24.2 s | 372 ms | 128 kB |

  Observation to keep in the paper: EZKL's prover cost is nearly flat across the
  Digits paths (it is set by `logrows`), so adaptive routing would save little
  there at this scale, while it grows strongly on the conv paths. These are CPU
  numbers; a CUDA-zkML vs EZKL ratio is a cross-hardware comparison and must be
  labelled as such (`benchmarks/COMPARABILITY_PROTOCOL.md`).

## Risks to watch when running

* The conv p12 statement has 784 public inputs and 768 hidden units (dense
  lowered matrix with zero coefficients). It fits easily on Colab GPUs; on a
  4 GB card (GTX 1650) watch memory and start with `WORKLOADS=mnist_mlp`.
* `pcani_colab_experiment.py` aborts if measured prover medians are not
  non-decreasing across paths; rerun with more `REPEATS` on noisy GPUs.
* MNIST downloads from OpenML on first use (cached in `.cache/openml`).
