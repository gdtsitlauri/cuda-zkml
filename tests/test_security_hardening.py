from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_pk_v3_does_not_serialize_toxic_waste():
    src = (ROOT / "src" / "prover" / "groth16.cu").read_text(encoding="utf-8")
    save_block = src[src.index("bool ProvingKey::save"):src.index("bool ProvingKey::load_streaming")]
    assert "Saved hardened v3 proving key" in save_block
    assert "write_fr(debug_trapdoor" not in save_block
    assert "write_vec_fr(tau_powers_scalars" not in save_block
    assert "write_vec_fr(A_query_scalars" not in save_block
    assert "Refusing to serialize scalar/trapdoor material" in save_block


def test_legacy_secret_bearing_pk_is_rejected():
    src = (ROOT / "src" / "prover" / "groth16.cu").read_text(encoding="utf-8")
    load_block = src[src.index("bool ProvingKey::load_streaming"):]
    assert "Refusing legacy/unknown proving key" in load_block
    assert 'std::memcmp(magic, "ZKMLPK3", 7)' in load_block


def test_unsound_linear_aggregation_is_disabled():
    src = (ROOT / "src" / "prover" / "groth16.cu").read_text(encoding="utf-8")
    agg = src[src.index("Groth16Proof Groth16Prover::aggregate_proofs"):src.index("Groth16 Verification")]
    assert "DISABLED" in agg
    assert "return Groth16Proof();" in agg


def test_groth16_c_uses_consistent_unblinded_b1_convention():
    src = (ROOT / "src" / "prover" / "groth16.cu").read_text(encoding="utf-8")
    # This implementation forms B1 = beta + sum(w_i B_i) in G1 without the
    # s*delta term, so C = L + H + s*A + r*B1.  Adding an explicit
    # -r*s*delta term with that B1 convention would over-correct.
    assert "B1 = beta + sum(w_i B_i)" in src
    assert "Subtracting r*s*delta here would over-correct" in src
    assert "C_jac = C_jac + (-rs_delta)" not in src


def test_pcani_exact_model_statement_fixes_parameters_in_circuit():
    prove = (ROOT / "src" / "cli" / "prove.cu").read_text(encoding="utf-8")
    circuit = (ROOT / "src" / "prover" / "circuit.cuh").read_text(encoding="utf-8")
    assert "build_fixed_model_circuit" in prove
    assert "add_fixed_linear_layer" in prove
    assert "add_fixed_linear_layer" in circuit
    assert "fixed-quantized-parameters-in-r1cs" in prove


def test_integer_model_payload_fails_closed_on_trailing_values():
    inference = (ROOT / "src" / "nn" / "inference.cuh").read_text(encoding="utf-8")
    block = inference[inference.index("bool load_integer_weights"):inference.index("// Initialize with random weights")]
    assert "integer model payload has %d trailing values" in block
    assert "return false" in block


def test_public_input_json_no_longer_truncates_to_low_limb():
    prove = (ROOT / "src" / "cli" / "prove.cu").read_text(encoding="utf-8")
    assert "uint256_limbs_to_decimal" in prove
    assert "low-limb decimal for lightweight interoperability" not in prove


def test_signed_fp_to_fr_conversion_is_preserved_in_cli_and_witness():
    root = Path(__file__).resolve().parents[1]
    prove = (root / "src" / "cli" / "prove.cu").read_text(encoding="utf-8")
    witness = (root / "src" / "prover" / "witness.cuh").read_text(encoding="utf-8")
    assert "Preserve the signed integer meaning when moving Fp -> Fr" in prove
    assert "zkml::fp_to_int(v)" in prove
    assert witness.count("Preserve signed integer semantics across Fp and Fr") == 2
    assert witness.count("fp_to_int(v)") >= 2
