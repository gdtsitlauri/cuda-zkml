#include "prover/groth16.cuh"
#include "prover/circuit.cuh"
#include "common.cuh"
#include <cstdio>
#include <string>
#include <vector>

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) printf("  TEST: %-44s ", name)
#define PASS() do { printf("[PASS]\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

using namespace zkml;
using bn254::Fr;

static R1CS build_mul_circuit() {
    // Witness layout: [1, z(public), x(private), y(private)]
    R1CS c;
    c.num_variables = 4;
    c.num_public_inputs = 1;
    c.num_private_inputs = 2;

        // Dummy constraint to ensure w[0] participates (w[0] * 1 = w[0])
        c.add_mul_constraint(
            0, Fr::one(),
            0, Fr::one(),
            0, Fr::one()
        );

    c.add_mul_constraint(
        2, Fr::one(),
        3, Fr::one(),
        1, Fr::one()
    );

    return c;
}

static Witness build_mul_witness(uint64_t x_u64, uint64_t y_u64) {
    Witness w;
    Fr x = Fr::from_uint(x_u64);
    Fr y = Fr::from_uint(y_u64);
    Fr z = x * y;

    w.values.resize(4);
    w.values[0] = Fr::one();
    w.values[1] = z;
    w.values[2] = x;
    w.values[3] = y;

    w.num_public = 1;
    w.num_private = 2;
    return w;
}

void test_groth16_small() {

    R1CS circuit = build_mul_circuit();
    Witness witness = build_mul_witness(7, 9);
    printf("\n=== Groth16 Small Circuit ===\n");

    TEST("witness satisfies R1CS");
    CHECK(circuit.verify_witness(witness.values), "valid witness should satisfy circuit");

    ProvingKey pk;
    VerificationKey vk;


    TEST("trusted setup executes");
    Groth16Prover::setup(circuit, pk, vk);
    CHECK(pk.num_variables == circuit.num_variables, "pk metadata should match circuit");

    TEST("proof generation executes");
    Groth16Proof proof = Groth16Prover::prove(pk, circuit, witness);
    CHECK(proof.valid, "proof should be marked valid");

    TEST("proof verifies with correct public input");
    std::vector<Fr> pub = witness.get_public();
    bool ok = Groth16Verifier::verify(vk, proof, pub);
    CHECK(ok, "proof should verify");

    TEST("proof fails with wrong public input");
    std::vector<Fr> wrong_pub = pub;
    wrong_pub[0] = wrong_pub[0] + Fr::from_uint(1);
    bool bad = Groth16Verifier::verify(vk, proof, wrong_pub);
    CHECK(!bad, "proof should not verify with incorrect public input");

    const std::string pk_path = "test_groth16_roundtrip.pk";

    TEST("proving key save succeeds");
    bool saved = pk.save(pk_path.c_str());
    CHECK(saved, "pk.save should succeed");

    ProvingKey loaded_pk;
    TEST("proving key load succeeds");
    bool loaded = loaded_pk.load_streaming(pk_path.c_str());
    CHECK(loaded, "pk.load_streaming should succeed");

    TEST("loaded proving key preserves scalar query data");
    CHECK(!loaded_pk.A_query_scalars.empty() &&
          (int)loaded_pk.A_query_scalars.size() == circuit.num_variables &&
          loaded_pk.debug_trapdoor.available,
          "loaded pk should contain scalar queries and trapdoor metadata");

    TEST("proof generation works after PK reload");
    Groth16Proof proof_reloaded = Groth16Prover::prove(loaded_pk, circuit, witness);
    CHECK(proof_reloaded.valid, "proof from loaded pk should be valid");

    TEST("reloaded PK proof verifies");
    bool ok_reloaded = Groth16Verifier::verify(vk, proof_reloaded, pub);
    CHECK(ok_reloaded, "proof from loaded pk should verify");

    std::remove(pk_path.c_str());
}

int main() {
    printf("Groth16 Test Suite\n");
    printf("==================\n");

    test_groth16_small();

    printf("\n==================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
