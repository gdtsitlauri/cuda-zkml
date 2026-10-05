// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "./Verifier.sol";

/**
 * @title PCANIContextVerifier
 * @notice On-chain anti-replay for PCANI v2 proofs (roadmap 1.1).
 *
 * The v2 statement carries a public context bound inside the R1CS
 * (ctx * ctx = ctx_sq). This contract recomputes the context from its parts,
 * exactly as python/zkml/statement_v2.py::compute_context does:
 *
 *   ctx = SHA-256( for part in ["PCANI-CTX-v1", sessionId, nonce, uint64(timestamp), policyDigest]:
 *                    uint32_be(len(part)) || part )  mod 2^253
 *
 * and accepts a proof only if (1) the context public input equals it, (2) the
 * timestamp is within the freshness window, (3) the context was never used
 * before, and (4) the Groth16 proof verifies.
 */
contract PCANIContextVerifier {
    Groth16Verifier public immutable verifier;
    uint256 public immutable contextIndex;      // public-input index of the context (statement metadata)
    uint256 public immutable freshnessWindow;   // seconds
    mapping(uint256 => bool) public usedContext;

    event ProofAccepted(uint256 indexed context, bytes32 indexed sessionId, address submitter);

    constructor(Groth16Verifier _verifier, uint256 _contextIndex, uint256 _freshnessWindow) {
        verifier = _verifier;
        contextIndex = _contextIndex;
        freshnessWindow = _freshnessWindow;
    }

    function computeContext(
        bytes memory sessionId,
        bytes memory nonce,
        uint64 timestamp,
        bytes memory policyDigest
    ) public pure returns (uint256) {
        bytes memory tag = "PCANI-CTX-v1";
        bytes32 h = sha256(abi.encodePacked(
            uint32(tag.length), tag,
            uint32(sessionId.length), sessionId,
            uint32(nonce.length), nonce,
            uint32(8), timestamp,
            uint32(policyDigest.length), policyDigest
        ));
        return uint256(h) & ((uint256(1) << 253) - 1);
    }

    function verifyWithContext(
        Groth16Verifier.Proof memory proof,
        uint256[] memory publicInputs,
        bytes memory sessionId,
        bytes memory nonce,
        uint64 timestamp,
        bytes memory policyDigest
    ) external returns (bool) {
        require(contextIndex < publicInputs.length, "context index out of range");
        require(timestamp <= block.timestamp && block.timestamp - timestamp <= freshnessWindow, "stale context");
        uint256 ctx = computeContext(sessionId, nonce, timestamp, policyDigest);
        require(publicInputs[contextIndex] == ctx, "context mismatch");
        require(!usedContext[ctx], "context already used");
        require(verifier.verifyProof(proof, publicInputs), "invalid proof");
        usedContext[ctx] = true;
        emit ProofAccepted(ctx, keccak256(sessionId), msg.sender);
        return true;
    }
}
