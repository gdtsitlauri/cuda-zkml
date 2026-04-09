// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

/**
 * @title Groth16 Verifier for CUDA-zkML
 * @notice Verifies Groth16 proofs of neural network inference on-chain.
 *
 * This contract verifies that a neural network was correctly executed
 * on given inputs WITHOUT revealing the model weights or private inputs.
 *
 * Verification equation:
 *   e(A, B) == e(α, β) · e(vk_x, γ) · e(C, δ)
 *
 * where vk_x = IC[0] + Σ public_inputs[i] * IC[i+1]
 *
 * Uses the BN254 (alt_bn128) precompiles available on Ethereum:
 *   - ecAdd (0x06): Point addition on G1
 *   - ecMul (0x07): Scalar multiplication on G1
 *   - ecPairing (0x08): Bilinear pairing check
 *
 * Gas cost estimate: ~250,000 gas for verification (mostly pairing)
 */
contract Groth16Verifier {

    // BN254 curve order
    uint256 constant PRIME_Q = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

    // Verification key (set during deployment)
    struct VerifyingKey {
        // G1 points
        uint256[2] alpha;    // α ∈ G1
        // G2 points (each coordinate is Fp2 = [real, imag])
        uint256[2][2] beta;  // β ∈ G2
        uint256[2][2] gamma; // γ ∈ G2
        uint256[2][2] delta; // δ ∈ G2
        // IC points for public inputs
        uint256[2][] ic;     // IC[0..num_public] ∈ G1
    }

    struct Proof {
        uint256[2] a;       // A ∈ G1
        uint256[2][2] b;    // B ∈ G2
        uint256[2] c;       // C ∈ G1
    }

    VerifyingKey internal vk;
    address public owner;

    event ProofVerified(address indexed prover, bool valid);

    constructor() {
        owner = msg.sender;
    }

    /**
     * @notice Set the verification key (only owner)
     * @dev In production, the VK would be set once during deployment
     */
    function setVerifyingKey(
        uint256[2] memory _alpha,
        uint256[2][2] memory _beta,
        uint256[2][2] memory _gamma,
        uint256[2][2] memory _delta,
        uint256[2][] memory _ic
    ) external {
        require(msg.sender == owner, "Only owner");
        require(_ic.length > 0, "IC must not be empty");

        vk.alpha = _alpha;
        vk.beta = _beta;
        vk.gamma = _gamma;
        vk.delta = _delta;

        delete vk.ic;
        for (uint256 i = 0; i < _ic.length; i++) {
            vk.ic.push(_ic[i]);
        }
    }

    /**
     * @notice Verify a Groth16 proof
     * @param proof The proof (A, B, C)
     * @param publicInputs The public inputs to the circuit
     * @return True if the proof is valid
     */
    function verifyProof(
        Proof memory proof,
        uint256[] memory publicInputs
    ) public returns (bool) {
        require(publicInputs.length + 1 == vk.ic.length, "Wrong number of public inputs");

        // Compute vk_x = IC[0] + Σ publicInputs[i] * IC[i+1]
        uint256[2] memory vk_x = vk.ic[0];

        for (uint256 i = 0; i < publicInputs.length; i++) {
            require(publicInputs[i] < PRIME_Q, "Public input exceeds field");

            // IC[i+1] * publicInputs[i]
            uint256[2] memory term = ecMul(vk.ic[i + 1], publicInputs[i]);

            // vk_x = vk_x + term
            vk_x = ecAdd(vk_x, term);
        }

        // Verify pairing equation:
        // e(-A, B) * e(α, β) * e(vk_x, γ) * e(C, δ) == 1
        //
        // This is checked via the ecPairing precompile which verifies:
        // Σ e(P_i, Q_i) == 1

        // Negate A (negate y coordinate)
        uint256[2] memory neg_a;
        neg_a[0] = proof.a[0];
        neg_a[1] = (PRIME_Q - proof.a[1]) % PRIME_Q;

        // Build pairing input: 4 pairs of (G1, G2) points
        uint256[24] memory input;

        // Pair 1: e(-A, B)
        // Ethereum BN254 precompiles expect Fp2 coordinates as [imag, real].
        input[0] = neg_a[0];
        input[1] = neg_a[1];
        input[2] = proof.b[0][1];
        input[3] = proof.b[0][0];
        input[4] = proof.b[1][1];
        input[5] = proof.b[1][0];

        // Pair 2: e(α, β)
        input[6] = vk.alpha[0];
        input[7] = vk.alpha[1];
        input[8] = vk.beta[0][1];
        input[9] = vk.beta[0][0];
        input[10] = vk.beta[1][1];
        input[11] = vk.beta[1][0];

        // Pair 3: e(vk_x, γ)
        input[12] = vk_x[0];
        input[13] = vk_x[1];
        input[14] = vk.gamma[0][1];
        input[15] = vk.gamma[0][0];
        input[16] = vk.gamma[1][1];
        input[17] = vk.gamma[1][0];

        // Pair 4: e(C, δ)
        input[18] = proof.c[0];
        input[19] = proof.c[1];
        input[20] = vk.delta[0][1];
        input[21] = vk.delta[0][0];
        input[22] = vk.delta[1][1];
        input[23] = vk.delta[1][0];

        // Call pairing precompile
        uint256[1] memory result;
        bool success;

        assembly {
            // ecPairing precompile at address 0x08
            // Input: 24 * 32 = 768 bytes (4 pairs × 6 uint256)
            // Output: 1 uint256 (1 if valid, 0 if invalid)
            success := staticcall(
                sub(gas(), 2000),  // gas
                0x08,              // ecPairing precompile
                input,             // input pointer
                768,               // input size (24 * 32)
                result,            // output pointer
                32                 // output size
            )
        }

        require(success, "Pairing check call failed");

        bool valid = result[0] == 1;
        emit ProofVerified(msg.sender, valid);

        return valid;
    }

    /**
     * @notice View function for proof verification (no state changes)
     */
    function verifyProofView(
        Proof memory proof,
        uint256[] memory publicInputs
    ) external view returns (bool) {
        require(publicInputs.length + 1 == vk.ic.length, "Wrong number of public inputs");

        uint256[2] memory vk_x = vk.ic[0];
        for (uint256 i = 0; i < publicInputs.length; i++) {
            require(publicInputs[i] < PRIME_Q, "Public input exceeds field");
            vk_x = ecAdd(vk_x, ecMul(vk.ic[i + 1], publicInputs[i]));
        }

        uint256[2] memory neg_a;
        neg_a[0] = proof.a[0];
        neg_a[1] = (PRIME_Q - proof.a[1]) % PRIME_Q;

        uint256[24] memory input;
        input[0] = neg_a[0]; input[1] = neg_a[1];
        input[2] = proof.b[0][1]; input[3] = proof.b[0][0];
        input[4] = proof.b[1][1]; input[5] = proof.b[1][0];
        input[6] = vk.alpha[0]; input[7] = vk.alpha[1];
        input[8] = vk.beta[0][1]; input[9] = vk.beta[0][0];
        input[10] = vk.beta[1][1]; input[11] = vk.beta[1][0];
        input[12] = vk_x[0]; input[13] = vk_x[1];
        input[14] = vk.gamma[0][1]; input[15] = vk.gamma[0][0];
        input[16] = vk.gamma[1][1]; input[17] = vk.gamma[1][0];
        input[18] = proof.c[0]; input[19] = proof.c[1];
        input[20] = vk.delta[0][1]; input[21] = vk.delta[0][0];
        input[22] = vk.delta[1][1]; input[23] = vk.delta[1][0];

        uint256[1] memory result;
        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 0x08, input, 768, result, 32)
        }

        require(success, "Pairing failed");
        return result[0] == 1;
    }

    // ============================================================
    // BN254 Precompile Wrappers
    // ============================================================

    /// @dev G1 point addition using precompile at 0x06
    function ecAdd(uint256[2] memory p1, uint256[2] memory p2)
        internal view returns (uint256[2] memory r)
    {
        uint256[4] memory input;
        input[0] = p1[0]; input[1] = p1[1];
        input[2] = p2[0]; input[3] = p2[1];

        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 0x06, input, 128, r, 64)
        }
        require(success, "ecAdd failed");
    }

    /// @dev G1 scalar multiplication using precompile at 0x07
    function ecMul(uint256[2] memory p, uint256 s)
        internal view returns (uint256[2] memory r)
    {
        uint256[3] memory input;
        input[0] = p[0]; input[1] = p[1]; input[2] = s;

        bool success;
        assembly {
            success := staticcall(sub(gas(), 2000), 0x07, input, 96, r, 64)
        }
        require(success, "ecMul failed");
    }
}
