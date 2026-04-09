#include "curve/g1.cuh"
#include "curve/g2.cuh"
#include "curve/pairing.cuh"
#include "common.cuh"
#include <cstdio>

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) printf("  TEST: %-40s ", name)
#define PASS() do { printf("[PASS]\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

using namespace bn254;

template <typename Point>
static Point scalar_mul_raw(const Point& base, const uint64_t scalar[4]) {
    Point result = Point::identity();
    int top_bit = -1;
    for (int i = 3; i >= 0; --i) {
        if (scalar[i] != 0) {
            uint64_t v = scalar[i];
            int clz = 0;
            while (((v >> (63 - clz)) & 1ULL) == 0) {
                clz++;
            }
            top_bit = i * 64 + (63 - clz);
            break;
        }
    }

    if (top_bit < 0) return result;

    for (int bit = top_bit; bit >= 0; --bit) {
        result = result.dbl();
        int limb = bit / 64;
        int pos = bit % 64;
        if ((scalar[limb] >> pos) & 1ULL) {
            result = result + base;
        }
    }
    return result;
}

void test_g1_basic() {
    printf("\n=== G1 Basic Operations ===\n");

    G1Affine gen = G1Affine::generator();

    TEST("G1 generator on curve");
    CHECK(gen.is_on_curve(), "generator should be on curve");

    TEST("G1 identity");
    G1Jacobian id = G1Jacobian::identity();
    CHECK(id.is_infinity(), "identity should be infinity");

    TEST("G1 generator + identity = generator");
    G1Jacobian gen_j = G1Jacobian::from_affine(gen);
    G1Jacobian sum = gen_j + id;
    G1Affine sum_a = sum.to_affine();
    CHECK(sum_a == gen, "G + O should be G");

    TEST("G1 point doubling");
    G1Jacobian doubled = gen_j.dbl();
    CHECK(!doubled.is_infinity(), "2G should not be infinity");
    G1Affine doubled_a = doubled.to_affine();
    CHECK(doubled_a.is_on_curve(), "2G should be on curve");

    TEST("G1 addition: G + G = 2G");
    G1Jacobian added = gen_j + gen_j;
    G1Affine added_a = added.to_affine();
    CHECK(added_a == doubled_a, "G + G should equal 2G");

    TEST("G1 scalar mul: 1 * G = G");
    Fr one = Fr::from_uint(1);
    G1Jacobian one_g = gen_j.scalar_mul(one);
    G1Affine one_g_a = one_g.to_affine();
    CHECK(one_g_a == gen, "1 * G should be G");

    TEST("G1 scalar mul: 2 * G = G + G");
    Fr two = Fr::from_uint(2);
    G1Jacobian two_g = gen_j.scalar_mul(two);
    G1Affine two_g_a = two_g.to_affine();
    CHECK(two_g_a == doubled_a, "2G via scalar_mul should equal G.dbl()");

    TEST("G1 scalar mul: 3 * G");
    Fr three = Fr::from_uint(3);
    G1Jacobian three_g = gen_j.scalar_mul(three);
    G1Jacobian expected_3g = doubled + gen_j;
    G1Affine three_a = three_g.to_affine();
    G1Affine exp_3a = expected_3g.to_affine();
    CHECK(three_a == exp_3a, "3G should be consistent");

    TEST("G1 negation: G + (-G) = O");
    G1Jacobian neg = -gen_j;
    G1Jacobian should_id = gen_j + neg;
    CHECK(should_id.is_infinity(), "G + (-G) should be identity");

    TEST("G1 scalar mul: larger scalar (100)");
    Fr hundred = Fr::from_uint(100);
    G1Jacobian hun_g = gen_j.scalar_mul(hundred);
    CHECK(!hun_g.is_infinity(), "100G should not be infinity");
    G1Affine hun_a = hun_g.to_affine();
    CHECK(hun_a.is_on_curve(), "100G should be on curve");
}

void test_g2_basic() {
    printf("\n=== G2 Basic Operations ===\n");

    G2Affine gen = G2Affine::generator();
    G2Jacobian gen_j = G2Jacobian::from_affine(gen);
    const uint64_t r_modulus[4] = {
        0x43E1F593F0000001ULL,
        0x2833E84879B97091ULL,
        0xB85045B68181585DULL,
        0x30644E72E131A029ULL
    };

    TEST("G2 identity");
    G2Jacobian id = G2Jacobian::identity();
    CHECK(id.is_infinity(), "G2 identity should be infinity");

    TEST("G2 generator + identity");
    G2Jacobian sum = gen_j + id;
    G2Affine sum_a = sum.to_affine();
    CHECK(sum_a == gen, "G2: G + O should be G");

    TEST("G2 doubling");
    G2Jacobian doubled = gen_j.dbl();
    CHECK(!doubled.is_infinity(), "G2: 2G should not be infinity");

    TEST("G2 addition G + G = 2G");
    G2Jacobian added = gen_j + gen_j;
    CHECK(added == doubled, "G2: G + G should equal 2G");

    TEST("G2 scalar mul: 1 * G = G");
    Fr one = Fr::from_uint(1);
    G2Jacobian one_g = gen_j.scalar_mul(one);
    G2Affine one_a = one_g.to_affine();
    CHECK(one_a == gen, "G2: 1G should be G");

    TEST("G2 negation: G + (-G) = O");
    G2Jacobian neg = -gen_j;
    G2Jacobian should_id = gen_j + neg;
    CHECK(should_id.is_infinity(), "G2: G + (-G) should be identity");

    TEST("G2 generator is in r-torsion subgroup");
    G2Jacobian r_times_g = scalar_mul_raw(gen_j, r_modulus);
    CHECK(r_times_g.is_infinity(), "G2 generator should satisfy r*G = O");
}

// GPU kernel test for curve operations
__global__ void test_g1_gpu_kernel(G1Affine* results) {
    if (threadIdx.x != 0) return;

    G1Affine gen = G1Affine::generator();
    G1Jacobian gen_j = G1Jacobian::from_affine(gen);

    // 5G
    Fr five = Fr::from_uint(5);
    G1Jacobian five_g = gen_j.scalar_mul(five);
    results[0] = five_g.to_affine();

    // 5G via repeated addition
    G1Jacobian acc = G1Jacobian::identity();
    for (int i = 0; i < 5; i++) {
        acc = acc + gen_j;
    }
    results[1] = acc.to_affine();
}

void test_g1_gpu() {
    printf("\n=== G1 GPU Tests ===\n");

    G1Affine* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, 2 * sizeof(G1Affine)));
    test_g1_gpu_kernel<<<1, 1>>>(d_results);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    G1Affine h_results[2];
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 2 * sizeof(G1Affine),
                           cudaMemcpyDeviceToHost));

    TEST("GPU: 5G scalar_mul == 5G addition");
    CHECK(h_results[0] == h_results[1], "GPU 5G methods should agree");

    TEST("GPU: 5G on curve");
    CHECK(h_results[0].is_on_curve(), "GPU 5G should be on curve");

    CUDA_CHECK(cudaFree(d_results));
}

void test_pairing_basic() {
    printf("\n=== Pairing Basic Tests ===\n");

    G1Affine P = G1Affine::generator();
    G2Affine Q = G2Affine::generator();
    const uint64_t r_modulus[4] = {
        0x43E1F593F0000001ULL,
        0x2833E84879B97091ULL,
        0xB85045B68181585DULL,
        0x30644E72E131A029ULL
    };

    TEST("pairing(P, infinity) = 1");
    G2Affine inf_q = G2Jacobian::identity().to_affine();
    CHECK(pairing(P, inf_q) == Fp12::one(), "pairing with infinity in G2 should be one");

    TEST("pairing(infinity, Q) = 1");
    G1Affine inf_p = G1Jacobian::identity().to_affine();
    CHECK(pairing(inf_p, Q) == Fp12::one(), "pairing with infinity in G1 should be one");

    TEST("e(P,Q) * e(-P,Q) = 1");
    G1Affine negP(P.x, -P.y);
    G1Affine Ps[2] = {P, negP};
    G2Affine Qs[2] = {Q, Q};
    CHECK(verify_pairing_product(Ps, Qs, 2), "pairing product inverse relation should hold");

    TEST("pairing bilinearity in G1: e(2P,Q)=e(P,Q)^2");
    G1Affine twoP = G1Jacobian::from_affine(P).dbl().to_affine();
    Fp12 ePQ = pairing(P, Q);
    Fp12 e2PQ = pairing(twoP, Q);
    CHECK(e2PQ == ePQ.sqr(), "pairing should be bilinear in G1");

    TEST("pairing bilinearity in G2: e(P,2Q)=e(P,Q)^2");
    G2Affine twoQ = G2Jacobian::from_affine(Q).dbl().to_affine();
    Fp12 eP2Q = pairing(P, twoQ);
    CHECK(eP2Q == ePQ.sqr(), "pairing should be bilinear in G2");

    TEST("pairing final result lies in GT subgroup");
    CHECK(ePQ.pow_vartime(r_modulus) == Fp12::one(), "pairing(P,Q)^r should be one");
}

int main() {
    printf("BN254 Curve Operations Test Suite\n");
    printf("=================================\n");

    test_g1_basic();
    test_g2_basic();
    test_g1_gpu();
    test_pairing_basic();

    printf("\n=================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
