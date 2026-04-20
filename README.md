# CUDA-zkML: GPU-Accelerated Zero-Knowledge Proofs for Neural Network Inference

**Author:** George David Tsitlauri  
**Affiliation:** Dept. of Informatics & Telecommunications, University of Thessaly, Greece  
**Contact:** gdtsitlauri@gmail.com  
**Year:** 2026

A CUDA-native system that generates zero-knowledge proofs for neural network inference. Proves that a model ran correctly on given inputs **without revealing the model weights or inputs**, using GPU parallelism for significant speedup over CPU-based tools.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Model       │────>│  Quantizer   │────>│  Field       │
│  (ONNX/PT)   │     │  float→Fp    │     │  Inference   │
└──────────────┘     └──────────────┘     │  (GPU)       │
                                           └──────┬───────┘
┌──────────────┐                                   │
│  Input Data  │───────────────────────────────────┘
└──────────────┘                                   │
                     ┌──────────────┐     ┌────────v───────┐
                     │  R1CS        │<────│  Witness       │
                     │  Circuit     │     │  Generator     │
                     └──────┬───────┘     └────────────────┘
                            │
                     ┌──────v───────┐
                     │  Groth16     │
                     │  Prover      │
                     │  (GPU: MSM   │
                     │   + NTT)     │
                     └──────┬───────┘
                            │
                     ┌──────v───────┐     ┌──────────────┐
                     │  Proof       │────>│  Verifier    │
                     │  (A, B, C)   │     │  (Native +   │
                     │  ~256 bytes  │     │   Solidity)  │
                     └──────────────┘     └──────────────┘
```

## Hardware Requirements

| Component | Minimum | Tested |
|-----------|---------|--------|
| GPU | NVIDIA (Compute ≥ 7.5) | GTX 1650 (4GB) |
| VRAM | 4 GB | 4 GB |
| CUDA | 12.0+ | 12.x |
| RAM | 8 GB | 16 GB |
| OS | Windows 10+ / Linux | Windows 11, Ubuntu 22.04 |

**Note:** Optimized for GTX 1650 (SM 7.5, 896 CUDA cores, 4GB VRAM).
All GPU allocations fit within a 3.5GB budget. For GPUs with more VRAM,
larger models and batch sizes are automatically supported.

## Quick Start

### Build (Linux)

```bash
# Prerequisites: CUDA 12.x, CMake 3.18+, GCC 11+
sudo apt install cmake build-essential

# Clone and build
git clone https://github.com/cuda-zkml/cuda-zkml.git
cd cuda-zkml
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Build (Windows)

```powershell
# Prerequisites: CUDA 12.x, CMake 3.18+, Visual Studio 2022
# MSVC host compiler with CUDA toolkit integration

mkdir build && cd build
cmake .. -G "Visual Studio 17 2022"
cmake --build . --config Release
```

### Run Demo

```bash
# Generate and verify a proof for MNIST MLP inference
./zkml-prove --demo
```

Expected output:
```
=== CUDA-zkML Demo: Proving MNIST MLP Inference ===
GPU: NVIDIA GeForce GTX 1650 (SM 7.5, 16 SMs, 4096 MB VRAM)
[1/5] Creating MNIST MLP model (784 → 128 → 10)...
[2/5] Generating random input (784 pixels)...
[3/5] Running quantized inference on GPU...
[4/5] Generating witness and R1CS circuit...
[5/5] Running Groth16 proof generation (GPU-accelerated)...
=== Results ===
Proof valid: YES
PROOF GENERATION COMPLETE.
```

### Prove Custom Model

```bash
# Prove inference on a custom model
./zkml-prove --model model.onnx --input input.npy --output proof.bin

# Verify the proof
./zkml-verify --vk vk.bin --proof proof.bin --public-inputs inputs.json
```

Notes:
- `zkml-prove` accepts model files in `.bin` and `.onnx` formats.
- `zkml-prove` accepts input files in `.bin` and `.npy` (`float32`) formats.
- `zkml-verify` accepts public inputs in `.bin` and `.json` formats.
- ONNX conversion now emits `model.weights.bin.arch.json` sidecars so the CLI can
  reconstruct non-MNIST feedforward topologies without hardcoded dimensions.
- For raw `.bin` weights, pass `--arch model.bin.arch.json` or place the sidecar
  next to the weights so `zkml-prove` can auto-detect it.
- The built-in defaults cover:
  - MNIST MLP: `784 -> 128 -> 10`
  - CIFAR-10 flattened MLP: `3072 -> 256 -> 64 -> 10`
  - Tiny transformer: `32 -> self_attention(4x8) -> ReLU -> 4`
- The CIFAR-10 flattened MLP path has been smoke-tested through
  `zkml-prove -> zkml-verify` using architecture sidecars on the GTX 1650 setup.
  The tiny transformer attention path now also passes the same CLI
  `zkml-prove -> zkml-verify` flow end-to-end.
  Architecture sidecars now also support stacked multi-head attention blocks
  (`num_heads` metadata on `self_attention` layers), and the repository test
  suite exercises a 2-head / 2-block transformer prove/verify path.

### Reuse a Proving Key

```bash
# First run: setup + save PK
./zkml-prove --model model.bin --input input.bin --output proof.bin \
  --vk vk.bin --public-inputs public_inputs.bin --pk-save pk.bin

# Second run: skip setup and reuse PK
./zkml-prove --model model.bin --input input2.bin --output proof2.bin \
  --vk vk.bin --public-inputs public_inputs2.bin --pk-load pk.bin
```

Notes:
- `--pk-save` stores a proving key with scalar query data for later reuse.
- `--pk-load` requires an existing `vk.bin` for verification and artifact checks.
- Large proving keys automatically stay in streaming mode and only materialize the
  scalar-domain queries needed by the current proving path.

### Python API

```python
from zkml import Prover, Model, export_solidity_bundle
import numpy as np

# Load model
model = Model.from_onnx("model.onnx")
# Or: model = Model.create_mnist_mlp()
# Or: model = Model.create_cifar_mlp()

# Create prover
prover = Prover(model, cli_path="./build")

# Prove and verify
input_data = np.random.randn(model.input_size).astype(np.float32)
result = prover.prove_and_verify(input_data)

print(f"Valid: {result['valid']}")
print(f"Prove time: {result['prove_time_ms']:.1f} ms")
print(f"Proof size: {result['proof_size_bytes']} bytes")

# Export Solidity/Ethers-friendly calldata after proving
bundle = export_solidity_bundle(
    "vk.bin",
    "proof.bin",
    "public_inputs.bin",
    "solidity_artifacts"
)
print(bundle["verifyProofArgs"]["publicInputs"])
```

### Export Solidity Artifacts

```bash
cd python
python export_solidity.py \
  --vk ../build/vk.bin \
  --proof ../build/proof.bin \
  --public-inputs ../build/public_inputs.bin \
  --output-dir ../build/solidity_artifacts
```

Generated files:
- `vk.solidity.json`
- `proof.solidity.json`
- `public_inputs.solidity.json`
- `calldata.solidity.json`
- `verifier_call.txt`

### Local On-Chain Verification Workflow

```bash
cd python
python onchain_verify.py \
  --vk ../proofs/vk.bin \
  --proof ../proofs/proof.bin \
  --public-inputs ../proofs/public_inputs.bin \
  --output-json ../proofs/onchain_report.json \
  --output-dir ../proofs/solidity_artifacts
```

What this does:
- compiles `contracts/Verifier.sol` with `solc`,
- deploys it to a local `eth-tester` / `py-evm` chain,
- uploads the verification key,
- runs `verifyProofView(...)`,
- writes a structured report with gas usage and validation status.

The repository test suite now covers this local EVM flow end-to-end.

## Project Structure

```
cuda-zkml/
├── src/
│   ├── common.cuh              # CUDA macros, memory helpers, uint256_t
│   ├── field/                   # BN254 finite field arithmetic
│   │   ├── fp.cuh/cu           # Fp (254-bit prime field) + Fr (scalar field)
│   │   ├── fp2.cuh             # Fp2 = Fp[u]/(u²+1) extension
│   │   ├── fp6.cuh             # Fp6 = Fp2[v]/(v³-ξ) extension
│   │   ├── fp12.cuh            # Fp12 = Fp6[w]/(w²-v), Frobenius, final exp
│   │   ├── montgomery.cuh      # CIOS Montgomery mul (PTX + MSVC intrinsics)
│   │   └── uint128_compat.cuh  # Portable 128-bit ops (MSVC / GCC / device)
│   ├── curve/                   # Elliptic curve operations
│   │   ├── g1.cuh/cu           # G1 Jacobian (BN254 y²=x³+3)
│   │   ├── g2.cuh/cu           # G2 Jacobian (twist curve over Fp2)
│   │   └── pairing.cuh         # Optimal Ate pairing, Miller loop, final exp
│   ├── ntt/                     # NTT with twiddle precomputation
│   │   └── ntt.cuh/cu          # Gentleman-Sande DIF + cooperative groups
│   ├── msm/                     # Pippenger MSM with warp-level primitives
│   │   └── msm.cuh/cu          # G1 + G2 MSM, streaming for >1M points
│   ├── nn/                      # Neural network inference in finite field
│   │   ├── quantize.cuh        # Float → Fp quantization (symmetric int8/16)
│   │   ├── layers.cuh/cu       # MatMul, ReLU, Softmax in Fp
│   │   └── inference.cuh/cu    # Full model inference + trace recording
│   ├── prover/                  # Groth16 zk-SNARK prover
│   │   ├── circuit.cuh         # R1CS constraint system + CircuitBuilder
│   │   ├── witness.cuh         # Witness generation from inference trace
│   │   └── groth16.cuh/cu      # QAP, setup, prove, KZG, batch, aggregate
│   ├── verifier/                # Proof verification + file I/O
│   │   └── verifier.cuh/cu     # Proof/VK serialization
│   └── cli/                     # CLI tools
│       ├── prove.cu             # zkml-prove (--demo, --model, --input)
│       └── verify.cu            # zkml-verify (--vk, --proof, --public-inputs)
├── contracts/
│   └── Verifier.sol             # Solidity Groth16 verifier (BN254 precompiles)
├── python/                      # Python API + benchmarks
│   ├── zkml/                    # Python package (model.py, prover.py)
│   │   └── artifacts.py         # proof/vk parsing + Solidity export helpers
│   ├── quantize.py              # Standalone quantization script
│   ├── export_solidity.py       # CLI exporter for Solidity/Ethers calldata
│   ├── onchain_verify.py        # Local EVM deploy + verify harness
│   ├── run_ezkl_benchmark.py    # Live EZKL benchmark helper
│   ├── run_orion_benchmark.py   # Orion toolchain probe / adapter
│   └── benchmark.py             # Measured local benchmarks + live/external baselines
├── benchmarks/
│   ├── baselines/               # Optional EZKL/Orion measured results
│   │   └── README.md            # External baseline JSON schema
│   └── results/                 # Generated JSON + markdown benchmark outputs
├── tests/                       # Test suite (CUDA + Python)
│   ├── test_field.cu            # BN254 Fp/Fp2/Fp6/Fp12 tests
│   ├── test_curve.cu            # G1/G2 curve operations
│   ├── test_ntt.cu              # NTT correctness
│   ├── test_msm.cu              # Pippenger MSM tests
│   ├── test_nn.py               # Quantization + NN layer tests
│   ├── test_e2e.py              # End-to-end pipeline
│   ├── test_solidity_export.py  # Solidity artifact export coverage
│   └── test_onchain_workflow.py # Local EVM verification workflow
├── paper/
│   └── cuda_zkml.tex            # arxiv-ready LaTeX paper
└── README.md
```

## Testing

```bash
# Build and run all tests
cd build
make -j$(nproc)
ctest --verbose

# Individual tests
./test_field    # BN254 field arithmetic
./test_curve    # Elliptic curve operations
./test_ntt      # Number Theoretic Transform
./test_msm      # Multi-Scalar Multiplication

# Python tests
cd ../tests
python test_nn.py       # Quantization and NN tests
python test_e2e.py      # End-to-end pipeline
python test_solidity_export.py  # Solidity artifact export
```

### Faster Dev Loop (recommended while editing `groth16.cu`)

`groth16.cu` is the heaviest CUDA translation unit in this repo. For faster local iteration:

```powershell
# Configure once (Windows)
cmake -S . -B build-fast -G "Visual Studio 17 2022" `
       -DZKML_FAST_DEBUG_LOOP=ON `
       -DZKML_FAST_ITERATION=ON `
       -DZKML_ULTRA_FAST_COMPILE=ON `
       -DZKML_NVCC_THREADS=1 `
       -DZKML_NVCC_SPLIT_COMPILE_THREADS=1

# Rebuild only what you need
cmake --build build-fast --config Release --target zkml_prover -- /m:1
cmake --build build-fast --config Release --target test_groth16 -- /m:1
```

Notes:
- `ZKML_ULTRA_FAST_COMPILE=ON` uses lower optimization and disables RDC for prover/CLI targets to reduce compile latency.
- `ZKML_NVCC_THREADS` and `ZKML_NVCC_SPLIT_COMPILE_THREADS` cap compiler parallelism (`1` is safest on thermally-limited laptops).
- `tests/test_ntt.cu` and `tests/test_msm.cu` now skip heavy benchmarks by default.
- To run benchmarks explicitly:
       - `test_ntt --bench`
       - `test_msm --bench`
       - or set `ZKML_RUN_BENCHMARKS=1`
- If you already have a working build directory with an explicit CUDA toolset, reuse the same toolset in `build-fast`.
- Keep `/m:1` for low-stress builds on laptops; increase only if the machine is stable.

## Benchmarks

```bash
cd python
python benchmark.py --runs 1 --live-ezkl --live-orion
```

The benchmark runner now measures:
- local `zkml-prove --demo` timing on the current machine,
- local tiny-transformer `zkml-prove` timing on the current machine,
- local CPU float32 and quantized-field inference baselines,
- live EZKL benchmark stages via `python/run_ezkl_benchmark.py`,
- live Orion toolchain probing via `python/run_orion_benchmark.py`,
- or optional EZKL / Orion rows loaded from `benchmarks/baselines/*.json`.

Current benchmark status on the tested GTX 1650 laptop:
- CUDA-zkML local demo is fully measured and working.
- EZKL now has a measured live row from WSL Ubuntu on the same machine.
- The current native Windows EZKL wheel still panics during `setup(...)`, so the
  repository treats WSL/Linux as the reliable EZKL path on this setup.
- Orion now has a measured WSL Ubuntu row from a passing archived `relu_i8`
  filtered test target.
- The full archived Orion workspace test suite still has upstream failures on
  this setup, so the repository uses a passing filtered target instead of the
  failing full-suite run.

Generated benchmark artifacts:
- `benchmarks/results/benchmark_results.json`
- `benchmarks/results/benchmark_results.md`
- `benchmarks/results/onchain_report.json`
- optional external baselines can be provided as `benchmarks/baselines/ezkl.json`
  and `benchmarks/baselines/orion.json`

Latest local measured snapshot on the tested laptop:

| System | Status | Inference (ms) | Setup (ms) | Prove (ms) | Verify (ms) | Proof Size |
|--------|--------|----------------|------------|------------|-------------|------------|
| CUDA-zkML (GTX 1650) | measured | 53.66 | 1381.30 | 3982.29 | 225.18 | 256 B |
| CUDA-zkML Tiny Transformer (GTX 1650) | measured | 44.84 | 174.79 | 207.14 | 263.35 | 256 B |
| EZKL (WSL Ubuntu) | measured | 5.07 | 1547.85 | 1812.74 | 13.26 | 18209 B |
| Orion (WSL filtered) | measured | N/A | N/A | 456930.00 | N/A | N/A |

The local on-chain report under `benchmarks/results/onchain_report.json` currently
shows a successful local EVM verification with:
- `status = verified`
- `deploy_gas_used = 2115563`
- `set_vk_gas_used = 894575`

## Key Technical Details

### BN254 Field Arithmetic
- Montgomery multiplication using CIOS algorithm with PTX carry-chain assembly (sm_75)
- Portable host code: MSVC `_umul128`/`_addcarry_u64` intrinsics on Windows, `__int128` on Linux
- Full field tower: Fp → Fp2 → Fp6 → Fp12 with Frobenius endomorphisms
- 4 × uint64 limbs, 254-bit prime field

### Optimal Ate Pairing
- Complete Miller loop with proper doubling/addition steps and line function evaluation
- Q1/Q2 Frobenius correction steps for BN254 curve
- Hard part of final exponentiation using BN254 parameter x = 4965661367071055538
- Sparse Fp12 multiplication (`mul_by_024`) for efficient Miller loop

### Groth16 Prover
- Full R1CS → QAP transformation with Lagrange interpolation at τ
- Quotient polynomial h(x) via GPU NTT (polynomial multiply + divide)
- KZG polynomial commitment scheme (commit via MSM, open via synthetic division)
- Proper trusted setup with batch inversion for Lagrange denominators
- Real pairing-based verification: e(-A,B)·e(α,β)·e(vk_x,γ)·e(C,δ) == 1
- Proof aggregation via random linear combination with Fiat-Shamir challenges
- Streaming proving key loader for PK > 3GB

### MSM (Pippenger)
- Window-based bucket accumulation with configurable window size
- Cooperative groups: warp-level ballot for intra-warp collision detection
- Streaming mode for > 1M points (fits in 4GB VRAM)
- Warp-parallel bucket reduction with cross-stripe combination

### NTT
- Precomputed twiddle factors (eliminates per-thread exponentiation)
- Out-of-place with ping-pong buffers
- Cooperative groups: warp.sync() for small strides, block.sync() for large
- Shared memory optimization for intra-block stages
- Maximum size 2^24 (512M elements)

### Neural Network Inference
- Full inference in BN254 base field (ZK-provable)
- Symmetric int8/int16 quantization with scale tracking
- Polynomial ReLU approximation (degree 3, Horner's method)
- Taylor-series softmax approximation with field inversion
- Inference trace recording for automatic witness generation
- Feedforward architecture sidecars (`.arch.json`) for ONNX-converted models
- CIFAR-10 flattened MLP path through the same prover/verifier flow
- Tiny transformer self-attention path through the same prover/verifier flow
- Stacked multi-head self-attention path through architecture sidecars

### Solidity Verifier
- On-chain Groth16 verification using EVM BN254 precompiles
- ecAdd (0x06), ecMul (0x07), ecPairing (0x08)
- Local `eth-tester` deployment and verification harness included

## License

MIT License. See [LICENSE](LICENSE).

## Citation

```bibtex
@misc{tsitlauri2026cudazkml,
  author = {George David Tsitlauri},
  title  = {CUDA-zkML: GPU-Accelerated Zero-Knowledge Proofs for Neural Network Inference},
  year   = {2026},
  institution = {University of Thessaly},
  email  = {gdtsitlauri@gmail.com}
}
```
