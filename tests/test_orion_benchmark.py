#!/usr/bin/env python3

import json
import os
import subprocess
import sys
from pathlib import Path


def test_run_orion_benchmark_with_fake_scarb_repo(tmp_path):
    repo_dir = tmp_path / "orion"
    repo_dir.mkdir()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_impl = bin_dir / "fake_scarb.py"
    fake_impl.write_text(
        "import sys\n"
        "cmd = sys.argv[1] if len(sys.argv) > 1 else ''\n"
        "if cmd == 'build':\n"
        "    print('Finished release target(s) in 1 second')\n"
        "    raise SystemExit(0)\n"
        "if cmd == 'test':\n"
        "    print('testing fake_orion ...')\n"
        "    print('running 1 tests')\n"
        "    print('test fake::test ... ok (gas usage est.: 123456789)')\n"
        "    print('test result: ok. 1 passed; 0 failed; 0 ignored; 0 filtered out;')\n"
        "    raise SystemExit(0)\n"
        "if cmd == '--version':\n"
        "    print('scarb 2.5.3')\n"
        "    raise SystemExit(0)\n"
        "print('unknown scarb command')\n"
        "raise SystemExit(1)\n",
        encoding="utf-8",
    )
    scarb_cmd = bin_dir / "scarb.cmd"
    scarb_cmd.write_text(
        "@echo off\n"
        f"python \"{fake_impl}\" %*\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    env["PATH"] = str(bin_dir) + os.pathsep + env.get("PATH", "")

    script_path = Path(__file__).resolve().parent.parent / "python" / "run_orion_benchmark.py"
    result = subprocess.run(
        [sys.executable, str(script_path), "--repo", str(repo_dir)],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )

    assert result.returncode == 0, result.stderr or result.stdout
    payload = json.loads(result.stdout)
    assert payload["system"] == "Orion"
    assert payload["status"] == "measured"
    assert payload["avg_setup_ms"] is not None
    assert payload["avg_prove_ms"] is not None
    assert payload["avg_verify_ms"] is None
    assert payload["gas_usage_estimates"] == [123456789]
    assert payload["repo_path"] == str(repo_dir)
