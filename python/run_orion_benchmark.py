#!/usr/bin/env python3
"""
Run or ingest an Orion benchmark row.

The repository does not vendor the Orion/Cairo toolchain, so this helper is
designed to do one of three things:
1. Load a measured Orion row from a JSON file.
2. Execute a user-provided external benchmark command that emits JSON.
3. Probe the local toolchain and report that Orion is unavailable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional


def load_json_row(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        row = json.load(f)
    row.setdefault("system", "Orion")
    row.setdefault("status", "measured")
    return row


def probe_tool(name: str) -> Optional[str]:
    resolved = shutil.which(name)
    if resolved:
        return resolved

    candidates = [
        os.path.expanduser(os.path.join("~", ".local", "bin", name)),
        os.path.expanduser(os.path.join("~", ".cargo", "bin", name)),
    ]
    if os.name == "nt":
        candidates.extend(
            [
                os.path.expanduser(os.path.join("~", ".local", "bin", f"{name}.cmd")),
                os.path.expanduser(os.path.join("~", ".local", "bin", f"{name}.exe")),
                os.path.expanduser(os.path.join("~", ".cargo", "bin", f"{name}.cmd")),
                os.path.expanduser(os.path.join("~", ".cargo", "bin", f"{name}.exe")),
            ]
        )

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return None


def parse_gas_usage(stdout: str) -> List[int]:
    return [int(match) for match in re.findall(r"gas usage est\.: ([0-9]+)", stdout)]


def detect_orion_repo(explicit_repo: Optional[str]) -> Optional[str]:
    candidates = [
        explicit_repo,
        os.environ.get("ORION_REPO"),
        os.path.expanduser("~/orion"),
    ]
    for candidate in candidates:
        if not candidate:
            continue
        repo = os.path.abspath(os.path.expanduser(candidate))
        if os.path.isdir(repo):
            return repo
    return None


def run_scarb(repo_path: str, args: List[str], timeout: int = 1800) -> tuple[subprocess.CompletedProcess[str], float]:
    scarb_exe = probe_tool("scarb") or "scarb"
    start = time.perf_counter()
    result = subprocess.run(
        [scarb_exe, *args],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return result, elapsed_ms


def benchmark_local_orion_repo(
    repo_path: str,
    package: Optional[str] = None,
    test_filter: Optional[str] = None,
) -> Dict[str, Any]:
    build_result, build_ms = run_scarb(repo_path, ["build"])
    if build_result.returncode != 0:
        return {
            "system": "Orion",
            "status": "error",
            "avg_inference_ms": None,
            "avg_setup_ms": None,
            "avg_prove_ms": None,
            "avg_verify_ms": None,
            "avg_total_s": None,
            "proof_size_bytes": None,
            "hardware": "CPU",
            "note": f"scarb build failed in {repo_path}: {(build_result.stderr or build_result.stdout).strip()[:240]}",
        }

    test_args = ["test"]
    if package:
        test_args.extend(["-p", package])
    if test_filter:
        test_args.extend(["--", "-f", test_filter])

    test_result, test_ms = run_scarb(repo_path, test_args)
    if test_result.returncode != 0:
        return {
            "system": "Orion",
            "status": "error",
            "avg_inference_ms": None,
            "avg_setup_ms": build_ms,
            "avg_prove_ms": None,
            "avg_verify_ms": None,
            "avg_total_s": build_ms / 1000.0,
            "proof_size_bytes": None,
            "hardware": "CPU",
            "note": f"scarb test failed in {repo_path}: {(test_result.stderr or test_result.stdout).strip()[:240]}",
            "stage_ms": {
                "build_ms": build_ms,
            },
        }

    row: Dict[str, Any] = {
        "system": "Orion",
        "model": "Archived Orion repo self-test suite" if not test_filter else f"Archived Orion filtered test ({test_filter})",
        "status": "measured",
        "avg_inference_ms": None,
        "avg_setup_ms": build_ms,
        "avg_prove_ms": test_ms,
        "avg_verify_ms": None,
        "avg_total_s": (build_ms + test_ms) / 1000.0,
        "proof_size_bytes": None,
        "hardware": "CPU",
        "note": (
            "Measured from scarb build + scarb test in a local Orion checkout; "
            "the Orion CLI path does not expose separate prove/verify timings here, "
            "so the test wall-clock is reported in the prove column."
        ),
        "stage_ms": {
            "build_ms": build_ms,
            "test_ms": test_ms,
        },
        "repo_path": repo_path,
    }
    if package:
        row["package"] = package
    if test_filter:
        row["test_filter"] = test_filter
    gas_estimates = parse_gas_usage(test_result.stdout)
    if gas_estimates:
        row["gas_usage_estimates"] = gas_estimates
    return row


def run_external_command(command: str, timeout: int = 900) -> Dict[str, Any]:
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=timeout,
        shell=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or result.stdout.strip() or f"command exited with {result.returncode}")

    stdout = result.stdout.strip()
    if not stdout:
        raise RuntimeError("command produced no JSON output")
    return json.loads(stdout)


def run_orion_live(
    command: Optional[str],
    json_path: Optional[str],
    repo_path: Optional[str],
    package: Optional[str],
    test_filter: Optional[str],
) -> Dict[str, Any]:
    if json_path and os.path.exists(json_path):
        row = load_json_row(json_path)
        row.setdefault("note", f"Loaded from {json_path}")
        return row

    if command:
        row = run_external_command(command)
        row.setdefault("system", "Orion")
        row.setdefault("status", "measured")
        row.setdefault("note", "Measured via external Orion benchmark command")
        return row

    detected_repo = detect_orion_repo(repo_path)
    if detected_repo and probe_tool("scarb"):
        return benchmark_local_orion_repo(detected_repo, package=package, test_filter=test_filter)

    scarb = probe_tool("scarb")
    cargo = probe_tool("cargo")
    toolchain = []
    if scarb:
        toolchain.append(f"scarb={scarb}")
    if cargo:
        toolchain.append(f"cargo={cargo}")

    note = "Orion toolchain not found"
    if toolchain:
        note = f"Orion benchmark command not provided; detected {'; '.join(toolchain)}"

    return {
        "system": "Orion",
        "status": "unavailable",
        "avg_inference_ms": None,
        "avg_setup_ms": None,
        "avg_prove_ms": None,
        "avg_verify_ms": None,
        "avg_total_s": None,
        "proof_size_bytes": None,
        "hardware": "CPU",
        "note": note,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or ingest an Orion benchmark row")
    parser.add_argument("--command", default=os.environ.get("ORION_BENCHMARK_CMD"))
    parser.add_argument("--json", default=os.environ.get("ORION_BENCHMARK_JSON"))
    parser.add_argument("--repo", default=os.environ.get("ORION_REPO"))
    parser.add_argument("--package", default=os.environ.get("ORION_PACKAGE"))
    parser.add_argument("--filter", dest="test_filter", default=os.environ.get("ORION_TEST_FILTER"))
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    row = run_orion_live(args.command, args.json, args.repo, args.package, args.test_filter)
    payload = json.dumps(row, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(payload)
            f.write("\n")
    print(payload)
    return 0 if row.get("status") == "measured" else 1


if __name__ == "__main__":
    raise SystemExit(main())
