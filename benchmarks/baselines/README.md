# External Baseline Inputs

Drop measured EZKL and Orion results in this directory if you want
`python/benchmark.py` to include them in the generated tables.

If you prefer live local probes instead of static JSON files:
- `python python/benchmark.py --live-ezkl`
- `python python/benchmark.py --live-orion`

The EZKL helper exercises a tiny reference ONNX model locally.
The Orion helper either runs a user-provided external command or reports
that the Cairo/Scarb toolchain is unavailable.

Supported filenames:
- `ezkl.json`
- `orion.json`

Expected JSON schema:

```json
{
  "system": "EZKL",
  "model": "MNIST MLP (784→128→10)",
  "status": "measured",
  "avg_inference_ms": null,
  "avg_setup_ms": null,
  "avg_prove_ms": 15000.0,
  "avg_verify_ms": 5.0,
  "avg_total_s": 15.0,
  "proof_size_bytes": 1500,
  "hardware": "Ryzen 5 5600H",
  "note": "Measured on local machine"
}
```

Any missing numeric field can be set to `null`.
