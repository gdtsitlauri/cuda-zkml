# Benchmark Comparability Protocol

A speedup ratio is reportable only when both systems share the same `workload_id` and the following fields match:

- model architecture and parameterization;
- input shape / batch size;
- quantization or fixed-point precision;
- public/private statement semantics;
- proof security target and whether model identity is bound;
- whether trusted setup is included in the reported stage;
- number of warmups and measured runs;
- hardware class, or an explicit statement that the comparison is cross-hardware.

Historical rows shipped with the repository predate this protocol and are **not comparable to one another**. They remain useful as local engineering measurements.

Recommended workload ID format:

`<dataset>-<architecture>-<precision>-<statement>-<batch>`

Example:

`mnist-mlp784x128x10-int8-private-input-fixed-model-b1`
