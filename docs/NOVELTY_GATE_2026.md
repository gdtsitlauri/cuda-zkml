# Novelty Gate — September 2026

This document is deliberately conservative. It does **not** claim that PCANI is novel until a systematic literature and patent review is complete.

## Nearby prior work already known

- **ZKML: An Optimizing System for ML Inference in Zero-Knowledge Proofs** (EuroSys 2024) optimizes ML-to-ZK circuit layout and demonstrates realistic models.
- **NANOZK: Layerwise Zero-Knowledge Proofs for Verifiable Large Language Model Inference** (2026) uses layerwise proofs and explores budgeted / Fisher-guided verification.
- **zkNAS: Secure and Efficient Outsourced-NAS with Zero-Cost Proxies** (DASFAA 2026) connects neural architecture search and zero-knowledge-oriented computation.
- The broader early-exit / conditional-computation literature already optimizes predictive compute, latency, or energy.

## Candidate gap worth testing

The narrow hypothesis for PCANI is:

> Jointly optimize *per-input routing* for **measured cryptographic proving cost**, then cryptographically prove the routing decision and the complete selected path with model/input linkage.

That combination appears narrower than ordinary early exit, circuit-layout optimization, layerwise proving, or NAS. However, "appears" is not enough for a novelty claim.

## Mandatory novelty check before publication

Search at minimum:

- arXiv, IACR ePrint, IEEE Xplore, ACM DL, Springer, USENIX;
- ZK/cryptography venues and ML systems/security venues;
- Google Patents / Espacenet for adaptive or conditional proof generation;
- codebases for EZKL, Orion/Giza, ICICLE-based zkML, Halo2 ML systems, zkVM/IVC ML inference.

Queries should combine terms such as:

- `zero knowledge adaptive inference`
- `zkML early exit`
- `conditional computation proof`
- `proof-aware neural routing`
- `dynamic neural network zero knowledge`
- `verifiable routing neural network`
- `input adaptive zkSNARK inference`
- `proof cost neural architecture`

## Go / no-go rule

Proceed as a novelty claim only if the exact contribution remains distinct after the audit. If a close match exists, narrow or change the contribution rather than overstating novelty.
