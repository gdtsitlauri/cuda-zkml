# CUDA-zkML Benchmark Results

| System | Status | Inference (ms) | Setup (ms) | Prove (ms) | Verify (ms) | Proof Size | Hardware | Note |
|--------|--------|----------------|------------|------------|-------------|------------|----------|------|
| CUDA-zkML (GTX 1650) | measured | 53.66 | 1381.30 | 3982.29 | 225.18 | 256 B | NVIDIA GTX 1650 (4GB, SM 7.5) | Measured from zkml-prove --demo on the local machine |
| CUDA-zkML Tiny Transformer (GTX 1650) | measured | 44.84 | 174.79 | 207.14 | 263.35 | 256 B | NVIDIA GTX 1650 (4GB, SM 7.5) | Measured from zkml-prove on the tiny transformer attention path |
| CPU Float32 Inference | measured | 0.09 | N/A | N/A | N/A | 0 B | AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD | Reference float32 forward pass only |
| CPU Quantized Field Inference | measured | 1.09 | N/A | N/A | N/A | 0 B | AMD64 Family 25 Model 80 Stepping 0, AuthenticAMD | Inference-only CPU baseline; float/quantized top-1 agreement 100.0% |
| EZKL | measured | 5.07 | 1547.85 | 1812.74 | 13.26 | 18209 B | CPU | Live local EZKL run in WSL Ubuntu on a tiny reference ONNX model |
| Orion | measured | N/A | N/A | 456930.00 | N/A | N/A | CPU | Measured in WSL Ubuntu from `scarb test -- -f relu_i8` on Orion v0.2.5; the archived Orion test workflow does not expose separate setup or verify timings. |

*Rows include the MNIST demo, tiny transformer attention path, local CPU baselines, and optional external comparisons.*
*Date: 2026-04-09*
