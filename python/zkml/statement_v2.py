"""PCANI statement v2 helpers (roadmap 1.1, 2.1-2.4, 2.6).

* compute_context: anti-replay context H(session_id || nonce || timestamp || policy_digest),
  SHA-256 truncated to 253 bits (always < r, same reduction as zkml-prove --context).
* load_statement_v2: parse zkml-prove --statement-v2 metadata + public inputs.
* model_commitment / input_commitment: Poseidon commitments (python/zkml/poseidon.py),
  identical to the in-circuit commitments of src/prover/statement_v2.cuh.
* reference_scores: exact integer inference for LINEAR / CONV2D / RELU_EXACT /
  RELU_APPROX layers -- the semantics the R1CS enforces.
"""
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import poseidon

R = poseidon.P
MASK253 = (1 << 253) - 1


def compute_context(session_id: str, nonce: bytes | str, timestamp: int, policy_digest: str) -> int:
    if isinstance(nonce, str):
        nonce = nonce.encode()
    h = hashlib.sha256()
    for part in (b"PCANI-CTX-v1", session_id.encode(), nonce, struct.pack(">Q", int(timestamp)),
                 bytes.fromhex(policy_digest) if all(c in "0123456789abcdefABCDEF" for c in policy_digest)
                 and len(policy_digest) % 2 == 0 else policy_digest.encode()):
        h.update(struct.pack(">I", len(part)))
        h.update(part)
    return int.from_bytes(h.digest(), "big") & MASK253


def context_hex(ctx: int) -> str:
    return f"0x{ctx:064x}"


def to_signed(v: int) -> int:
    v %= R
    return v - R if v > R // 2 else v


def model_commitment(weight_blocks: Sequence[Sequence[int]]) -> int:
    """weight_blocks: per LINEAR/CONV2D layer, weights then bias, in layer order."""
    flat: List[int] = []
    for blk in weight_blocks:
        flat.extend(int(x) % R for x in blk)
    return poseidon.commit(flat, poseidon.DOMAIN_MODEL)


def input_commitment(x: Sequence[int], blinding: int) -> int:
    return poseidon.commit([int(v) % R for v in x] + [blinding % R], poseidon.DOMAIN_INPUT)


@dataclass
class StatementV2:
    meta: Dict[str, Any]
    public: List[int]
    outputs: List[List[int]] = field(default_factory=list)      # per sample (signed), empty if private
    model_tag: int = 0
    context: Optional[int] = None
    model_commitment: Optional[int] = None
    inputs: List[List[int]] = field(default_factory=list)       # per sample (signed), if public
    input_commitments: List[int] = field(default_factory=list)  # per sample, if private
    route: List[tuple] = field(default_factory=list)            # per sample (flag, class)


def load_public_inputs_any(path: str) -> List[int]:
    if path.endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [int(x) for x in data]
    from .pcani_protocol import load_public_inputs   # binary format of zkml-prove
    return [int(x) for x in load_public_inputs(path)]


def parse_statement_v2(meta: Dict[str, Any], public: List[int]) -> StatementV2:
    if meta.get("statement_mode") != "pcani-v2" or int(meta.get("statement_version", 0)) != 2:
        raise ValueError("not a PCANI v2 statement")
    if len(public) != int(meta["num_public"]):
        raise ValueError(f"public input length {len(public)} != {meta['num_public']}")
    if any(not 0 <= v < R for v in public):
        # x and x + r verify alike in Groth16; only canonical values may be compared or
        # remembered (otherwise context replay checks could be bypassed by aliasing)
        raise ValueError("non-canonical public input (>= r)")
    st = StatementV2(meta=meta, public=public)
    B, n_in, n_out = int(meta["batch"]), int(meta["input_size"]), int(meta["output_size"])
    o = int(meta["outputs_start_index"])
    if o >= 0:
        st.outputs = [[to_signed(public[o + b * n_out + i]) for i in range(n_out)] for b in range(B)]
    st.model_tag = public[int(meta["model_tag_public_index"])]
    if int(meta["context_public_index"]) >= 0:
        st.context = public[int(meta["context_public_index"])]
    if int(meta["model_commitment_index"]) >= 0:
        st.model_commitment = public[int(meta["model_commitment_index"])]
    s, slot = int(meta["inputs_start_index"]), int(meta["input_slot_size"])
    if meta["input_visibility"] == "public":
        st.inputs = [[to_signed(v) for v in public[s + b * slot: s + (b + 1) * slot]] for b in range(B)]
    else:
        st.input_commitments = [public[s + b] for b in range(B)]
    rs = int(meta["route_start_index"])
    if rs >= 0:
        st.route = [(public[rs + 2 * b], public[rs + 2 * b + 1]) for b in range(B)]
    if int(meta["model_tag64"], 16) != st.model_tag:
        raise ValueError("model tag in public inputs does not match metadata")
    return st


def load_statement_v2(meta_path: str, public_inputs_path: str) -> StatementV2:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    return parse_statement_v2(meta, load_public_inputs_any(public_inputs_path))


# ---------------- exact integer reference inference ----------------
def _conv2d(x: List[int], cfg: Dict[str, Any], w: List[int], b: List[int]) -> List[int]:
    ic, ih, iw = cfg["in_channels"], cfg["in_height"], cfg["in_width"]
    oc, k = cfg["out_channels"], cfg["kernel_size"]
    s, p = cfg.get("stride", 1), cfg.get("padding", 0)
    oh, ow = (ih + 2 * p - k) // s + 1, (iw + 2 * p - k) // s + 1
    out = []
    for o in range(oc):
        for oy in range(oh):
            for ox in range(ow):
                acc = b[o]
                for c in range(ic):
                    for ky in range(k):
                        for kx in range(k):
                            yy, xx = oy * s + ky - p, ox * s + kx - p
                            if 0 <= yy < ih and 0 <= xx < iw:
                                acc += w[((o * ic + c) * k + ky) * k + kx] * x[(c * ih + yy) * iw + xx]
                out.append(acc)
    return out


def reference_scores(layers: List[Dict[str, Any]], params: List[int], x: Sequence[int]) -> List[int]:
    """layers: architecture-json layer dicts; params: the int32 payload (weights, bias per layer)."""
    cur, off = [int(v) for v in x], 0
    for L in layers:
        t = L["type"].lower()
        if t == "linear":
            n_in, n_out = L["in_features"], L["out_features"]
            w = params[off:off + n_in * n_out]; off += n_in * n_out
            b = params[off:off + n_out]; off += n_out
            cur = [b[i] + sum(w[i * n_in + j] * cur[j] for j in range(n_in)) for i in range(n_out)]
        elif t in ("conv2d", "conv"):
            nw = L["out_channels"] * L["in_channels"] * L["kernel_size"] ** 2
            w = params[off:off + nw]; off += nw
            b = params[off:off + L["out_channels"]]; off += L["out_channels"]
            cur = _conv2d(cur, L, w, b)
        elif t == "relu_exact":
            bits, shift = L.get("bits", 32), L.get("shift", 0)
            lo, hi = -(1 << (bits - 1)), 1 << (bits - 1)
            if any(not lo <= v < hi for v in cur):
                raise ValueError(f"relu_exact input outside the {bits}-bit range")
            cur = [max(v, 0) >> shift for v in cur]
        elif t in ("relu", "relu_approx"):
            cur = [v * v + 64 * v + 64 for v in cur]       # default RELU_APPROX polynomial
        elif t in ("softmax", "softmax_approx"):
            break
        else:
            raise ValueError(f"unsupported layer type {t}")
    if off != len(params):
        raise ValueError("parameter payload size mismatch")
    return cur


def route_decision(scores: Sequence[int], threshold: int) -> tuple:
    """(flag, top1) exactly as the circuit computes it (ties -> lowest index)."""
    top = max(range(len(scores)), key=lambda i: (scores[i], -i))
    second = max(scores[i] for i in range(len(scores)) if i != top)
    return (1 if scores[top] - second >= threshold else 0, top)
