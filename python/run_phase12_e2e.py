#!/usr/bin/env python3
"""End-to-end GPU validation of roadmap phases 1-2 (fail-closed).

Requires the CUDA build (zkml-prove, zkml-verify, zkml-ceremony). Experiments:
  E1 trained real-ReLU digits model, context-bound proof (1.1, 2.1):
       scores == Python exact reference; context in the statement; zkml-verify ok;
       proof with an altered context / altered score is rejected by the verifier
  E2 native CONV2D + RELU_EXACT CNN (2.2): proof verifies, scores exact
  E3 batching (2.6): B = 1, 4, 8 samples in one proof, proving time per sample
  E4 private model + private input + in-circuit routing + private outputs (2.3, 2.4):
       C_model / C_input equal the Python Poseidon commitments, route flag/class
       equal the reference decision, proof verifies
  E5 MPC phase-2 ceremony (2.5): setup -> 2 contributions -> transcripts verify ->
       proof with the final key verifies with the final vk and NOT with the setup vk
  E6 underconstraint check (1.2) on the exported E1/E4 circuits
Writes <out>/phase12_summary.json; exits non-zero on any failed check.

  python python/run_phase12_e2e.py --build-dir build --out results/phase12/<run>
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import secrets
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tests"))

from zkml.statement_v2 import (  # noqa: E402
    R, compute_context, context_hex, input_commitment, load_statement_v2, model_commitment,
    reference_scores, route_decision,
)

CHECKS: list[dict] = []


def check(name: str, ok: bool, **info) -> bool:
    CHECKS.append({"check": name, "ok": bool(ok), **info})
    print(("[PASS] " if ok else "[FAIL] ") + name + (f"  {info}" if info else ""))
    return ok


def run(cmd: list[str], log: pathlib.Path) -> tuple[int, str]:
    t0 = time.time()
    p = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    out = p.stdout + p.stderr
    log.write_text(" ".join(map(str, cmd)) + "\n\n" + out, encoding="utf-8")
    return p.returncode, out + f"\n[wall {time.time() - t0:.2f}s]"


def prove_ms(out: str) -> float | None:
    m = re.search(r"Proving time: ([0-9.]+) ms", out)
    return float(m.group(1)) if m else None


def write_cnn(d: pathlib.Path, seed: int = 1) -> tuple[list, list]:
    rng = np.random.default_rng(seed)
    layers = [
        {"type": "conv2d", "in_features": 64, "out_features": 4 * 64, "in_channels": 1, "in_height": 8,
         "in_width": 8, "out_channels": 4, "kernel_size": 3, "stride": 1, "padding": 1},
        {"type": "relu_exact", "in_features": 256, "out_features": 256, "bits": 16, "shift": 3},
        {"type": "linear", "in_features": 256, "out_features": 10},
        {"type": "softmax", "in_features": 10, "out_features": 10},
    ]
    params = np.concatenate([rng.integers(-4, 5, 36), rng.integers(-8, 9, 4),
                             rng.integers(-3, 4, 2560), rng.integers(-16, 17, 10)]).astype(np.int32)
    d.mkdir(parents=True, exist_ok=True)
    params.tofile(d / "model.int32.bin")
    (d / "model.arch.json").write_text(json.dumps({"name": "digits_cnn_conv2d", "layers": layers}, indent=2))
    return layers, [int(v) for v in params]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-dir", default="build")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    B = pathlib.Path(a.build_dir).resolve()
    OUT = pathlib.Path(a.out).resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    prove, verify, cer = B / "zkml-prove", B / "zkml-verify", B / "zkml-ceremony"
    summary: dict = {"experiments": {}}

    # ---------------- E1 ----------------
    m1 = OUT / "relu_digits"
    subprocess.run([sys.executable, str(HERE / "export_trained_relu_model.py"), "--out", str(m1)], check=True)
    report = json.loads((m1 / "report.json").read_text())
    expected = json.loads((m1 / "expected_scores.json").read_text())
    layers = json.loads((m1 / "model.arch.json").read_text())["layers"]
    params = [int(v) for v in np.fromfile(m1 / "model.int32.bin", dtype=np.int32)]
    ctx = compute_context("phase12-session", secrets.token_bytes(16), int(time.time()), "00" * 32)
    e1 = OUT / "e1"
    e1.mkdir(exist_ok=True)
    base = [prove, "--model", m1 / "model.int32.bin", "--arch", m1 / "model.arch.json", "--integer-model",
            "--integer-input", "--statement-v2"]
    rc, out = run(base + ["--input", m1 / "inputs/sample_0.npy", "--context", context_hex(ctx),
                          "--output", e1 / "proof.bin", "--vk", e1 / "vk.bin", "--public-inputs", e1 / "public.json",
                          "--statement-meta", e1 / "meta.json", "--export-r1cs", e1 / "stmt.r1cs",
                          "--export-wtns", e1 / "stmt.wtns", "--pk-save", e1 / "pk0.bin"], e1 / "prove.log")
    check("E1 prove (trained ReLU, context)", rc == 0, prove_ms=prove_ms(out))
    st = load_statement_v2(str(e1 / "meta.json"), str(e1 / "public.json"))
    check("E1 scores == Python exact reference", st.outputs[0] == expected["sample_0"]["scores"],
          scores=st.outputs[0])
    check("E1 context bound in statement", st.context == ctx)
    rc, _ = run([verify, "--vk", e1 / "vk.bin", "--proof", e1 / "proof.bin", "--public-inputs", e1 / "public.json"],
                e1 / "verify.log")
    check("E1 zkml-verify accepts", rc == 0)
    pub = json.loads((e1 / "public.json").read_text())
    for name, idx in (("context", int(st.meta["context_public_index"])), ("score", 0)):
        bad = list(pub)
        bad[idx] = str((int(bad[idx]) + 1) % R)
        (e1 / f"public_bad_{name}.json").write_text(json.dumps([int(x) for x in bad]))
        rc, _ = run([verify, "--vk", e1 / "vk.bin", "--proof", e1 / "proof.bin", "--public-inputs",
                     e1 / f"public_bad_{name}.json"], e1 / f"verify_bad_{name}.log")
        check(f"E1 verifier rejects altered {name}", rc != 0)
    summary["experiments"]["E1_trained_relu"] = {"model": report, "constraints": st.meta["r1cs_constraints"]}

    # ---------------- E2 ----------------
    m2 = OUT / "cnn"
    cnn_layers, cnn_params = write_cnn(m2)
    x0 = np.load(m1 / "inputs/sample_0.npy")
    e2 = OUT / "e2"
    e2.mkdir(exist_ok=True)
    rc, out = run([prove, "--model", m2 / "model.int32.bin", "--arch", m2 / "model.arch.json", "--integer-model",
                   "--integer-input", "--statement-v2", "--input", m1 / "inputs/sample_0.npy",
                   "--output", e2 / "proof.bin", "--vk", e2 / "vk.bin", "--public-inputs", e2 / "public.json",
                   "--statement-meta", e2 / "meta.json"], e2 / "prove.log")
    check("E2 prove (native CONV2D + RELU_EXACT)", rc == 0, prove_ms=prove_ms(out))
    st2 = load_statement_v2(str(e2 / "meta.json"), str(e2 / "public.json"))
    check("E2 scores == Python exact reference", st2.outputs[0] == reference_scores(cnn_layers, cnn_params, x0.astype(int).tolist()))
    rc, _ = run([verify, "--vk", e2 / "vk.bin", "--proof", e2 / "proof.bin", "--public-inputs", e2 / "public.json"],
                e2 / "verify.log")
    check("E2 zkml-verify accepts", rc == 0)
    dense_terms = 256 * (64 + 1)
    summary["experiments"]["E2_conv2d"] = {"constraints": st2.meta["r1cs_constraints"],
                                           "dense_lowering_linear_terms": dense_terms}

    # ---------------- E3 batching ----------------
    e3 = OUT / "e3"
    e3.mkdir(exist_ok=True)
    batch_res = {}
    for nb in (1, 4, 8):
        files = ",".join(str(m1 / f"inputs/sample_{i}.npy") for i in range(nb))
        rc, out = run(base + ["--batch-inputs", files, "--output", e3 / f"proof_{nb}.bin", "--vk", e3 / f"vk_{nb}.bin",
                              "--public-inputs", e3 / f"public_{nb}.json", "--statement-meta", e3 / f"meta_{nb}.json"],
                      e3 / f"prove_{nb}.log")
        ok = rc == 0
        if ok:
            stb = load_statement_v2(str(e3 / f"meta_{nb}.json"), str(e3 / f"public_{nb}.json"))
            ok = all(stb.outputs[i] == expected[f"sample_{i}"]["scores"] for i in range(nb))
            batch_res[nb] = {"prove_ms": prove_ms(out), "constraints": stb.meta["r1cs_constraints"]}
        check(f"E3 batch of {nb}: proof + exact scores", ok, **batch_res.get(nb, {}))
    summary["experiments"]["E3_batching"] = batch_res

    # ---------------- E4 private model / input / routing ----------------
    e4 = OUT / "e4"
    e4.mkdir(exist_ok=True)
    blind = [secrets.randbits(240) for _ in range(2)]
    files = ",".join(str(m1 / f"inputs/sample_{i}.npy") for i in range(2))
    rc, out = run(base + ["--batch-inputs", files, "--private-model", "--private-input", "--private-outputs",
                          "--input-blinding", ",".join(hex(b) for b in blind), "--route-threshold", "200",
                          "--output", e4 / "proof.bin", "--vk", e4 / "vk.bin", "--public-inputs", e4 / "public.json",
                          "--statement-meta", e4 / "meta.json", "--export-r1cs", e4 / "stmt.r1cs",
                          "--export-wtns", e4 / "stmt.wtns"], e4 / "prove.log")
    check("E4 prove (private model + private input + routing)", rc == 0, prove_ms=prove_ms(out))
    st4 = load_statement_v2(str(e4 / "meta.json"), str(e4 / "public.json"))
    blocks = [params[0:64 * 32], params[64 * 32:64 * 32 + 32], params[64 * 32 + 32:64 * 32 + 32 + 320], params[-10:]]
    check("E4 C_model == Python Poseidon commitment", st4.model_commitment == model_commitment(blocks))
    ok_in, ok_route = True, True
    for i in range(2):
        x = np.load(m1 / f"inputs/sample_{i}.npy").astype(int).tolist()
        ok_in &= st4.input_commitments[i] == input_commitment(x, blind[i])
        ok_route &= tuple(st4.route[i]) == route_decision(expected[f"sample_{i}"]["scores"], 200)
    check("E4 C_input == Python commitment(x || r)", ok_in)
    check("E4 route flag/class == reference decision", ok_route, route=st4.route)
    check("E4 scores are not public", st4.outputs == [] and st4.inputs == [])
    rc, _ = run([verify, "--vk", e4 / "vk.bin", "--proof", e4 / "proof.bin", "--public-inputs", e4 / "public.json"],
                e4 / "verify.log")
    check("E4 zkml-verify accepts", rc == 0)
    summary["experiments"]["E4_private"] = {"constraints": st4.meta["r1cs_constraints"], "prove_ms": prove_ms(out)}

    # ---------------- E5 ceremony ----------------
    e5 = OUT / "e5"
    e5.mkdir(exist_ok=True)
    keys = [(e1 / "pk0.bin", e1 / "vk.bin")]
    for i in (1, 2):
        pk_in, vk_in = keys[-1]
        pk_o, vk_o, tr = e5 / f"pk{i}.bin", e5 / f"vk{i}.bin", e5 / f"t{i}.bin"
        rc, _ = run([cer, "contribute", "--pk-in", pk_in, "--vk-in", vk_in, "--pk-out", pk_o, "--vk-out", vk_o,
                     "--transcript", tr, "--entropy", secrets.token_hex(16)], e5 / f"contribute{i}.log")
        check(f"E5 contribution {i}", rc == 0)
        rc, _ = run([cer, "verify", "--pk-before", pk_in, "--pk-after", pk_o, "--vk-after", vk_o, "--transcript", tr],
                    e5 / f"verify_contribution{i}.log")
        check(f"E5 contribution {i} verifies", rc == 0)
        keys.append((pk_o, vk_o))
    import shutil
    shutil.copy(keys[-1][1], e5 / "vk_final.bin")
    rc, _ = run(base + ["--input", m1 / "inputs/sample_0.npy", "--context", context_hex(ctx), "--pk-load", keys[-1][0],
                        "--vk", e5 / "vk_final.bin", "--output", e5 / "proof.bin", "--public-inputs", e5 / "public.json"],
                e5 / "prove.log")
    check("E5 prove with ceremony key", rc == 0)
    rc, _ = run([verify, "--vk", e5 / "vk_final.bin", "--proof", e5 / "proof.bin", "--public-inputs", e5 / "public.json"],
                e5 / "verify_final.log")
    check("E5 proof verifies with the ceremony vk", rc == 0)
    rc, _ = run([verify, "--vk", e1 / "vk.bin", "--proof", e5 / "proof.bin", "--public-inputs", e5 / "public.json"],
                e5 / "verify_setup_vk.log")
    check("E5 proof does NOT verify with the original setup vk", rc != 0)

    # ---------------- E6 underconstraint check ----------------
    from test_underconstrained import check_circuit, load_r1cs, load_wtns  # noqa: E402
    uc = {}
    for name, d in (("E1", e1), ("E4", e4)):
        r = check_circuit(load_r1cs(d / "stmt.r1cs"), load_wtns(d / "stmt.wtns"))
        uc[name] = {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()}
        check(f"E6 {name} circuit fully constrained", not (r["unsatisfied"] or r["unused"] or r["unpinned"]
                                                            or r["accepted_random_corruptions"]), **uc[name])
    summary["experiments"]["E6_underconstrained"] = uc

    summary["checks"] = CHECKS
    summary["all_passed"] = all(c["ok"] for c in CHECKS)
    (OUT / "phase12_summary.json").write_text(json.dumps(summary, indent=2))
    print("[PHASE12 E2E PASS]" if summary["all_passed"] else "[PHASE12 E2E FAIL]")
    return 0 if summary["all_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
