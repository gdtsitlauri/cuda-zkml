// Host test of the Groth16 phase-2 ceremony (roadmap 2.5).
// Uses a synthetic proving key whose secrets are known to the test, so it can
// check the exact algebra: after contributions s1, s2 the key must equal the key
// for delta * s1 * s2 (L, H scaled by the inverse). Also: transcript and key
// file round trips, verification of an honest chain, and rejection of
// tampered contributions. End-to-end proving with a ceremony key runs on GPU
// in the Colab notebook.

#include "prover/ceremony.cuh"
#include <cstdio>

using namespace zkml;
using namespace zkml::ceremony;

static int g_fail = 0;
static void expect(bool c, const char* name) {
    printf("%s %s\n", c ? "[PASS]" : "[FAIL]", name);
    if (!c) g_fail++;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    printf("== MPC phase-2 ceremony host tests\n");
    const G1Affine g1 = G1Affine::generator();
    const G2Affine g2 = G2Affine::generator();
    const Fr delta = Fr::from_uint(987654321), alpha = Fr::from_uint(11), beta = Fr::from_uint(13), gamma = Fr::from_uint(17);

    PK pk;
    pk.num_constraints = 4; pk.num_variables = 6; pk.num_public = 2;
    pk.alpha_g1 = mul1(g1, alpha); pk.beta_g1 = mul1(g1, beta); pk.delta_g1 = mul1(g1, delta);
    pk.beta_g2 = mul2(g2, beta); pk.gamma_g2 = mul2(g2, gamma); pk.delta_g2 = mul2(g2, delta);
    std::vector<Fr> lsc = {Fr::from_uint(5), Fr::from_uint(7), Fr::from_uint(9)}, hsc = {Fr::from_uint(2), Fr::from_uint(3), Fr::from_uint(4), Fr::from_uint(6)};
    for (auto& x : lsc) pk.L.push_back(mul1(g1, x * delta.inv()));
    for (auto& x : hsc) pk.H.push_back(mul1(g1, x * delta.inv()));
    for (int i = 0; i < 6; i++) { pk.A.push_back(mul1(g1, Fr::from_uint(100 + i))); pk.B1.push_back(mul1(g1, Fr::from_uint(200 + i))); pk.B2.push_back(mul2(g2, Fr::from_uint(200 + i))); }
    pk.tau_g1 = {g1, mul1(g1, Fr::from_uint(3))};
    pk.tau_g2 = {g2, mul2(g2, Fr::from_uint(3))};
    VK vk;
    vk.num_public = 2; vk.alpha_g1 = pk.alpha_g1; vk.beta_g2 = pk.beta_g2; vk.gamma_g2 = pk.gamma_g2; vk.delta_g2 = pk.delta_g2;
    vk.ic = {mul1(g1, Fr::from_uint(1)), mul1(g1, Fr::from_uint(2)), mul1(g1, Fr::from_uint(3))};

    // file round trips
    save_pk((dir + "/mpc_pk0.bin").c_str(), pk);
    save_vk((dir + "/mpc_vk0.bin").c_str(), vk);
    PK pk0; VK vk0;
    expect(load_pk((dir + "/mpc_pk0.bin").c_str(), pk0) && load_vk((dir + "/mpc_vk0.bin").c_str(), vk0) &&
               eq1(pk0.delta_g1, pk.delta_g1) && eq2(pk0.delta_g2, pk.delta_g2) && pk0.L.size() == 3 &&
               eq1(pk0.H[3], pk.H[3]) && eq2(vk0.delta_g2, vk.delta_g2) && vk0.ic.size() == 3,
           "ZKMLPK3 / VK file round trip");

    // two contributions
    const Fr s1 = Fr::from_uint(0xABCDEF), s2 = Fr::from_uint(0x123457);
    PK pk1 = pk0; VK vk1 = vk0;
    Transcript t1 = contribute(pk1, vk1, s1, Fr::from_uint(777));
    PK pk2 = pk1; VK vk2 = vk1;
    Transcript t2 = contribute(pk2, vk2, s2, Fr::from_uint(888));
    save_transcript((dir + "/mpc_t1.bin").c_str(), t1);
    Transcript t1r;
    expect(load_transcript((dir + "/mpc_t1.bin").c_str(), t1r) && eq1(t1r.S1, t1.S1) && eq2(t1r.S2, t1.S2) && t1r.z == t1.z,
           "transcript round trip");

    const Fr d2 = delta * s1 * s2;
    bool algebra = eq1(pk2.delta_g1, mul1(g1, d2)) && eq2(pk2.delta_g2, mul2(g2, d2)) && eq2(vk2.delta_g2, pk2.delta_g2);
    for (size_t i = 0; i < lsc.size(); i++) algebra &= eq1(pk2.L[i], mul1(g1, lsc[i] * d2.inv()));
    for (size_t i = 0; i < hsc.size(); i++) algebra &= eq1(pk2.H[i], mul1(g1, hsc[i] * d2.inv()));
    expect(algebra, "after 2 contributions: delta = d*s1*s2, L and H scaled by its inverse (exact algebra)");

    VerifyReport r1 = verify(pk0, pk1, vk1, t1), r2 = verify(pk1, pk2, vk2, t2);
    expect(r1.ok() && r2.ok(), "honest contributions verify (PoK, pairings, batch L/H, unchanged elements)");

    // tampering
    PK bad = pk2; bad.L[1] = mul1(bad.L[1], Fr::from_uint(2));
    expect(!verify(pk1, bad, vk2, t2).ok(), "tampered L query is rejected");
    PK bad2 = pk2; bad2.delta_g1 = mul1(pk1.delta_g1, Fr::from_uint(5));   // delta not multiplied by s2
    expect(!verify(pk1, bad2, vk2, t2).ok(), "delta inconsistent with the transcript is rejected");
    PK bad3 = pk2; bad3.A[0] = mul1(bad3.A[0], Fr::from_uint(3));
    expect(!verify(pk1, bad3, vk2, t2).ok(), "modified A query (not allowed in phase 2) is rejected");
    Transcript forged = t2; forged.z = forged.z + Fr::one();
    expect(!verify(pk1, pk2, vk2, forged).ok(), "forged proof of knowledge is rejected");
    VK badvk = vk2; badvk.delta_g2 = vk1.delta_g2;
    expect(!verify(pk1, pk2, badvk, t2).ok(), "stale vk.delta_g2 is rejected");

    printf(g_fail == 0 ? "[HOST CEREMONY TESTS PASS]\n" : "[HOST CEREMONY TESTS FAIL] %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
