"""Underconstraint checks on exported PCANI circuits (roadmap 1.2).

Independent of the C++ prover: parses the iden3 .r1cs / .wtns files written by
src/prover/r1cs_export.cuh (zkml-prove --export-r1cs, or tests/host/test_statement_v2)
with pure-Python BN254 arithmetic and checks:

  1. the witness satisfies every constraint;
  2. every wire appears in at least one constraint (no free-floating wire);
  3. single-wire perturbation: w[i] + 1 (and w[i] - 1) violates some constraint
     for every wire i >= 1 -- no wire can be changed alone while the public
     inputs and all other wires stay fixed;
  4. negative witnesses: random corruptions of internal wires are rejected.

These checks catch the common bug class (a value computed in the witness but
not constrained). They are not a full uniqueness proof; for that, the same
.r1cs files can be given to an SMT-based tool such as Picus.

Run:  pytest tests/test_underconstrained.py   (env PCANI_R1CS_DIR = export directory)
"""
from __future__ import annotations

import os
import pathlib
import random
import struct

import pytest

R = 21888242871839275222246405745257275088548364400416034343698204186575808495617


def _read_sections(data: bytes, magic: bytes) -> dict[int, bytes]:
    assert data[:4] == magic, f"bad magic {data[:4]!r}"
    n_sections = struct.unpack_from("<I", data, 8)[0]
    off, secs = 12, {}
    for _ in range(n_sections):
        typ, size = struct.unpack_from("<IQ", data, off)
        off += 12
        secs[typ] = data[off:off + size]
        off += size
    return secs


def load_r1cs(path: pathlib.Path):
    secs = _read_sections(path.read_bytes(), b"r1cs")
    h = secs[1]
    n8 = struct.unpack_from("<I", h, 0)[0]
    prime = int.from_bytes(h[4:4 + n8], "little")
    n_wires, n_pub_out, n_pub_in, n_prv_in = struct.unpack_from("<IIII", h, 4 + n8)
    m = struct.unpack_from("<I", h, 4 + n8 + 16 + 8)[0]
    cons, off, c = [], 0, secs[2]
    for _ in range(m):
        row = []
        for _ in range(3):
            n = struct.unpack_from("<I", c, off)[0]
            off += 4
            lc = []
            for _ in range(n):
                w = struct.unpack_from("<I", c, off)[0]
                v = int.from_bytes(c[off + 4:off + 4 + n8], "little")
                off += 4 + n8
                lc.append((w, v))
            row.append(lc)
        cons.append(row)
    return {"prime": prime, "n_wires": n_wires, "n_pub": n_pub_out + n_pub_in, "n_prv_in": n_prv_in,
            "constraints": cons}


def load_wtns(path: pathlib.Path) -> list[int]:
    secs = _read_sections(path.read_bytes(), b"wtns")
    n8 = struct.unpack_from("<I", secs[1], 0)[0]
    n = struct.unpack_from("<I", secs[1], 4 + n8)[0]
    v = secs[2]
    return [int.from_bytes(v[i * n8:(i + 1) * n8], "little") for i in range(n)]


def _dot(lc, w):
    return sum(c * w[i] for i, c in lc) % R


def check_circuit(r1cs, w, rng_seed=1) -> dict:
    cons = r1cs["constraints"]
    assert r1cs["prime"] == R
    assert len(w) == r1cs["n_wires"] and w[0] == 1
    ev = [(_dot(a, w), _dot(b, w), _dot(c, w)) for a, b, c in cons]
    unsat = [k for k, (a, b, c) in enumerate(ev) if (a * b - c) % R]
    by_wire: dict[int, list[tuple[int, int, int]]] = {}
    for k, row in enumerate(cons):
        for mat, lc in enumerate(row):
            for i, coeff in lc:
                by_wire.setdefault(i, []).append((k, mat, coeff))
    unused = [i for i in range(1, len(w)) if i not in by_wire]
    unpinned = []
    for i in range(1, len(w)):
        hits = by_wire.get(i, [])
        for delta in (1, R - 1):
            rows: dict[int, list[int]] = {}
            for k, mat, coeff in hits:
                rows.setdefault(k, [ev[k][0], ev[k][1], ev[k][2]])
                rows[k][mat] = (rows[k][mat] + coeff * delta) % R
            if all((a * b - c) % R == 0 for a, b, c in rows.values()):
                unpinned.append(i)
                break
    rng = random.Random(rng_seed)
    internal = list(range(1 + r1cs["n_pub"], len(w)))
    accepted_corruptions = 0
    for _ in range(min(200, len(internal))):
        w2 = list(w)
        for i in rng.sample(internal, k=min(3, len(internal))):
            w2[i] = rng.randrange(R)
        if all((_dot(a, w2) * _dot(b, w2) - _dot(c, w2)) % R == 0 for a, b, c in cons):
            accepted_corruptions += 1
    return {"constraints": len(cons), "wires": len(w), "unsatisfied": unsat, "unused": unused,
            "unpinned": unpinned, "accepted_random_corruptions": accepted_corruptions}


def _export_dir() -> pathlib.Path | None:
    d = os.environ.get("PCANI_R1CS_DIR")
    return pathlib.Path(d) if d else None


CIRCUITS = ["stmt_v2_A", "stmt_v2_B"]


@pytest.mark.parametrize("name", CIRCUITS)
def test_exported_circuit_is_fully_constrained(name):
    d = _export_dir()
    if d is None or not (d / f"{name}.r1cs").exists():
        pytest.skip("set PCANI_R1CS_DIR to the directory written by tests/host/run_host_tests.sh")
    r1cs = load_r1cs(d / f"{name}.r1cs")
    w = load_wtns(d / f"{name}.wtns")
    res = check_circuit(r1cs, w)
    assert res["unsatisfied"] == [], res["unsatisfied"][:5]
    assert res["unused"] == [], res["unused"][:5]
    assert res["unpinned"] == [], res["unpinned"][:5]
    assert res["accepted_random_corruptions"] == 0


def test_detector_flags_a_deliberately_underconstrained_circuit(tmp_path):
    """Sanity check of the checker itself: x*y = z with an extra unconstrained wire."""
    # wires: 0=1, 1=x(pub), 2=y, 3=z, 4=free
    cons = [[[(1, 1)], [(2, 1)], [(3, 1)]]]
    r1cs = {"prime": R, "n_wires": 5, "n_pub": 1, "n_prv_in": 0, "constraints": cons}
    res = check_circuit(r1cs, [1, 3, 5, 15, 42])
    assert res["unused"] == [4] and 4 in res["unpinned"]
    # y and z alone cannot be moved, x*y=z pins them pairwise only through x
    assert 2 not in res["unpinned"] and 3 not in res["unpinned"]


if __name__ == "__main__":
    import json
    import sys
    d = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    ok = True
    for n in CIRCUITS:
        r = check_circuit(load_r1cs(d / f"{n}.r1cs"), load_wtns(d / f"{n}.wtns"))
        summary = {k: (len(v) if isinstance(v, list) else v) for k, v in r.items()}
        print(n, json.dumps(summary))
        ok &= not (r["unsatisfied"] or r["unused"] or r["unpinned"] or r["accepted_random_corruptions"])
    print("[UNDERCONSTRAINED CHECK PASS]" if ok else "[UNDERCONSTRAINED CHECK FAIL]")
    sys.exit(0 if ok else 1)
