#!/usr/bin/env python3
"""
Compile, deploy, and exercise the Solidity verifier against CUDA-zkML artifacts.

The script performs:
1. Artifact parsing and optional Solidity export.
2. EVM BN254 curve-compatibility preflight using py_ecc.
3. Local contract compilation via solc.
4. Local deployment to an eth-tester / PyEVM chain.
5. Verification-key upload and proof verification attempt.

The generated JSON report is intentionally explicit about failure modes so the
current EVM-compatibility state is reproducible instead of hidden behind manual
steps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from zkml.artifacts import (
    ProofArtifact,
    VerificationKeyArtifact,
    build_solidity_bundle,
    evm_curve_compatibility_report,
    export_solidity_bundle,
    load_public_inputs,
)


def repo_root() -> str:
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def default_contract_path() -> str:
    return os.path.join(repo_root(), "contracts", "Verifier.sol")


def detect_solc_binary() -> Optional[str]:
    candidates = [
        os.environ.get("SOLC_BINARY"),
        os.path.join(repo_root(), ".solcx", "solc-v0.8.20", "solc.exe"),
        os.path.join(repo_root(), ".solcx", "solc-v0.8.20", "solc"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def compile_verifier(contract_path: str, solc_binary: str) -> Dict[str, Any]:
    try:
        from solcx import compile_source
    except ImportError as exc:
        raise RuntimeError("pip install py-solc-x to compile the Solidity verifier") from exc

    with open(contract_path, "r", encoding="utf-8") as f:
        source = f.read()

    compiled = compile_source(
        source,
        output_values=["abi", "bin"],
        solc_binary=solc_binary,
    )
    for name, artifact in compiled.items():
        if name.endswith(":Groth16Verifier"):
            return artifact
    raise RuntimeError("Groth16Verifier contract not found in compile output")


def run_onchain_verification(
    vk_path: str,
    proof_path: str,
    public_inputs_path: str,
    contract_path: str,
    solc_binary: str,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    vk = VerificationKeyArtifact.from_file(vk_path)
    proof = ProofArtifact.from_file(proof_path)
    public_inputs = load_public_inputs(public_inputs_path)
    curve_report = evm_curve_compatibility_report(vk, proof)

    report: Dict[str, Any] = {
        "status": "initialized",
        "contract_path": os.path.abspath(contract_path),
        "solc_binary": os.path.abspath(solc_binary),
        "artifacts": {
            "vk": os.path.abspath(vk_path),
            "proof": os.path.abspath(proof_path),
            "public_inputs": os.path.abspath(public_inputs_path),
        },
        "curve_compatibility": curve_report,
        "steps": {
            "compiled": False,
            "deployed": False,
            "vk_uploaded": False,
            "verify_called": False,
        },
    }

    if output_dir:
        bundle = export_solidity_bundle(vk_path, proof_path, public_inputs_path, output_dir)
        report["solidity_export_dir"] = os.path.abspath(output_dir)
        report["solidity_bundle_preview"] = {
            "public_inputs": len(bundle["publicInputs"]),
            "ic_points": len(bundle["vk"]["ic"]),
        }
    else:
        bundle = build_solidity_bundle(vk, proof, public_inputs)

    if not curve_report["all_points_evm_compatible"]:
        report["preflight_warning"] = (
            "At least one proof/VK point is not on the EVM BN254 curve. "
            "Deployment can still proceed, but the pairing precompile is expected to fail."
        )

    try:
        compiled = compile_verifier(contract_path, solc_binary)
        report["steps"]["compiled"] = True
    except Exception as exc:
        report["status"] = "compile_failed"
        report["compile_error"] = str(exc)
        return report

    try:
        from eth_tester import EthereumTester, PyEVMBackend
        from web3 import EthereumTesterProvider, Web3
    except ImportError as exc:
        report["status"] = "runtime_missing"
        report["runtime_error"] = (
            "Missing on-chain test dependencies. Install web3, eth-tester, and py-evm."
        )
        report["runtime_import_error"] = str(exc)
        return report

    try:
        t0 = time.perf_counter()
        provider = EthereumTesterProvider(EthereumTester(backend=PyEVMBackend()))
        w3 = Web3(provider)
        account = w3.eth.accounts[0]
        contract = w3.eth.contract(abi=compiled["abi"], bytecode=compiled["bin"])
        tx_hash = contract.constructor().transact({"from": account})
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        verifier = w3.eth.contract(address=receipt.contractAddress, abi=compiled["abi"])
        report["deploy_ms"] = (time.perf_counter() - t0) * 1000.0
        report["deploy_gas_used"] = int(receipt.gasUsed)
        report["contract_address"] = receipt.contractAddress
        report["steps"]["deployed"] = True
    except Exception as exc:
        report["status"] = "deploy_failed"
        report["deploy_error"] = str(exc)
        return report

    vk_python = vk.to_python_dict()
    proof_python = proof.to_python_dict()

    try:
        t0 = time.perf_counter()
        tx_hash = verifier.functions.setVerifyingKey(
            vk_python["alpha"],
            vk_python["beta"],
            vk_python["gamma"],
            vk_python["delta"],
            vk_python["ic"],
        ).transact({"from": account})
        receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
        report["set_vk_ms"] = (time.perf_counter() - t0) * 1000.0
        report["set_vk_gas_used"] = int(receipt.gasUsed)
        report["steps"]["vk_uploaded"] = True
    except Exception as exc:
        report["status"] = "set_vk_failed"
        report["set_vk_error"] = str(exc)
        return report

    try:
        t0 = time.perf_counter()
        valid = verifier.functions.verifyProofView(
            proof_python,
            [int(value) for value in public_inputs],
        ).call({"from": account})
        report["verify_call_ms"] = (time.perf_counter() - t0) * 1000.0
        report["steps"]["verify_called"] = True
        report["evm_valid"] = bool(valid)
        report["status"] = "verified" if valid else "invalid"
    except Exception as exc:
        report["verify_call_ms"] = (time.perf_counter() - t0) * 1000.0
        report["steps"]["verify_called"] = True
        report["evm_valid"] = False
        report["status"] = "verification_failed"
        report["verify_error"] = str(exc)

    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local EVM verification for CUDA-zkML artifacts")
    parser.add_argument("--vk", required=True, help="Path to vk.bin")
    parser.add_argument("--proof", required=True, help="Path to proof.bin")
    parser.add_argument("--public-inputs", required=True, help="Path to public_inputs.bin or .json")
    parser.add_argument(
        "--contract",
        default=default_contract_path(),
        help="Path to Verifier.sol",
    )
    parser.add_argument(
        "--solc-binary",
        default=detect_solc_binary(),
        help="Path to the solc executable (defaults to repo-local .solcx install)",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="Optional path to write a structured JSON report",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for exported Solidity calldata artifacts",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero unless on-chain verification returns true",
    )
    args = parser.parse_args()

    if not args.solc_binary:
        print("No solc binary found. Pass --solc-binary or install one under .solcx.", file=sys.stderr)
        return 1

    report = run_onchain_verification(
        vk_path=args.vk,
        proof_path=args.proof,
        public_inputs_path=args.public_inputs,
        contract_path=args.contract,
        solc_binary=args.solc_binary,
        output_dir=args.output_dir,
    )

    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
            f.write("\n")

    print("On-chain verification report")
    print("===========================")
    print(f"status: {report['status']}")
    curve_ok = report["curve_compatibility"]["all_points_evm_compatible"]
    print(f"curve_compatible: {curve_ok}")
    if "contract_address" in report:
        print(f"contract: {report['contract_address']}")
    if "deploy_gas_used" in report:
        print(f"deploy_gas: {report['deploy_gas_used']}")
    if "set_vk_gas_used" in report:
        print(f"set_vk_gas: {report['set_vk_gas_used']}")
    if "evm_valid" in report:
        print(f"evm_valid: {report['evm_valid']}")
    if "preflight_warning" in report:
        print(f"warning: {report['preflight_warning']}")
    if "verify_error" in report:
        print(f"verify_error: {report['verify_error']}")
    if args.output_json:
        print(f"report_json: {os.path.abspath(args.output_json)}")
    if args.output_dir:
        print(f"solidity_artifacts: {os.path.abspath(args.output_dir)}")

    if args.strict:
        return 0 if report.get("status") == "verified" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
