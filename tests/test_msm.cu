#include "msm/msm.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <random>

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) printf("  TEST: %-40s ", name)
#define PASS() do { printf("[PASS]\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

using namespace bn254;

// CPU reference MSM (naive)
G1Jacobian cpu_msm(const G1Affine* points, const Fr* scalars, int n) {
    G1Jacobian result = G1Jacobian::identity();
    for (int i = 0; i < n; i++) {
        G1Jacobian pt = G1Jacobian::from_affine(points[i]);
        G1Jacobian contrib = pt.scalar_mul(scalars[i]);
        result = result + contrib;
    }
    return result;
}

G2Jacobian cpu_msm_g2(const G2Affine* points, const Fr* scalars, int n) {
    G2Jacobian result = G2Jacobian::identity();
    for (int i = 0; i < n; i++) {
        G2Jacobian pt = G2Jacobian::from_affine(points[i]);
        G2Jacobian contrib = pt.scalar_mul(scalars[i]);
        result = result + contrib;
    }
    return result;
}

void test_msm_small() {
    printf("\n=== MSM Small Tests ===\n");

    G1Affine gen = G1Affine::generator();

    // MSM with 1 point: s * G
    TEST("MSM n=1: 5 * G");
    G1Affine points[1] = {gen};
    Fr scalars[1] = {Fr::from_uint(5)};
    G1Jacobian result = msm_g1(points, scalars, 1);

    G1Jacobian expected = G1Jacobian::from_affine(gen).scalar_mul(Fr::from_uint(5));
    G1Affine result_a = result.to_affine();
    G1Affine expected_a = expected.to_affine();
    CHECK(result_a == expected_a, "MSM(1) should match scalar_mul");

    // MSM with 2 points: 3*G + 7*G = 10*G
    TEST("MSM n=2: 3G + 7G = 10G");
    G1Affine pts2[2] = {gen, gen};
    Fr sc2[2] = {Fr::from_uint(3), Fr::from_uint(7)};
    G1Jacobian result2 = msm_g1(pts2, sc2, 2);
    G1Jacobian expected2 = G1Jacobian::from_affine(gen).scalar_mul(Fr::from_uint(10));
    CHECK(result2.to_affine() == expected2.to_affine(), "3G+7G should be 10G");
}

void test_msm_medium() {
    printf("\n=== MSM Medium Tests ===\n");

    // Generate n distinct points: i*G for i = 1..n
    int n = 64;
    std::vector<G1Affine> points(n);
    std::vector<Fr> scalars(n);

    G1Jacobian gen_j = G1Jacobian::from_affine(G1Affine::generator());
    G1Jacobian acc = gen_j;
    for (int i = 0; i < n; i++) {
        points[i] = acc.to_affine();
        scalars[i] = Fr::from_uint((uint64_t)(i + 1));
        acc = acc + gen_j;
    }

    TEST("MSM n=64: GPU vs CPU reference");
    CudaTimer timer;

    timer.start();
    G1Jacobian gpu_result = msm_g1(points.data(), scalars.data(), n);
    float gpu_ms = timer.stop();

    timer.start();
    G1Jacobian cpu_result = cpu_msm(points.data(), scalars.data(), n);
    float cpu_ms = timer.stop();

    G1Affine gpu_a = gpu_result.to_affine();
    G1Affine cpu_a = cpu_result.to_affine();
    CHECK(gpu_a == cpu_a, "GPU MSM should match CPU MSM");
    printf("    GPU: %.2f ms, CPU: %.2f ms, Speedup: %.1fx\n",
           gpu_ms, cpu_ms, cpu_ms / gpu_ms);
}

void test_msm_g2() {
    printf("\n=== MSM G2 Tests ===\n");

    G2Affine gen = G2Affine::generator();

    TEST("MSM G2 n=1: 5 * G2");
    G2Affine points1[1] = {gen};
    Fr scalars1[1] = {Fr::from_uint(5)};
    G2Jacobian result1 = msm_g2(points1, scalars1, 1);
    G2Jacobian expected1 = G2Jacobian::from_affine(gen).scalar_mul(Fr::from_uint(5));
    CHECK(result1.to_affine() == expected1.to_affine(), "G2 MSM(1) should match scalar_mul");

    TEST("MSM G2 n=16: GPU vs CPU reference");
    int n = 16;
    std::vector<G2Affine> points(n);
    std::vector<Fr> scalars(n);

    G2Jacobian gen_j = G2Jacobian::from_affine(gen);
    G2Jacobian acc = gen_j;
    for (int i = 0; i < n; i++) {
        points[i] = acc.to_affine();
        scalars[i] = Fr::from_uint((uint64_t)(i + 3));
        acc = acc + gen_j;
    }

    G2Jacobian gpu_result = msm_g2(points.data(), scalars.data(), n);
    G2Jacobian cpu_result = cpu_msm_g2(points.data(), scalars.data(), n);
    CHECK(gpu_result.to_affine() == cpu_result.to_affine(), "GPU G2 MSM should match CPU MSM");
}

void test_msm_benchmark() {
    printf("\n=== MSM Benchmarks ===\n");

    G1Affine gen = G1Affine::generator();
    G1Jacobian gen_j = G1Jacobian::from_affine(gen);

    for (int log_n = 8; log_n <= 14; log_n += 2) {
        int n = 1 << log_n;

        // Generate points: precompute i*G
        std::vector<G1Affine> points(n);
        G1Jacobian acc = gen_j;
        for (int i = 0; i < n; i++) {
            points[i] = acc.to_affine();
            acc = acc + gen_j;
        }

        // Random scalars
        std::mt19937_64 rng(42);
        std::vector<Fr> scalars(n);
        for (int i = 0; i < n; i++) {
            scalars[i] = Fr::from_uint(rng() & 0xFFFFFFFF);
        }

        // Check VRAM
        size_t mem_needed = n * (sizeof(G1Affine) + sizeof(Fr)) +
                            (1 << 14) * sizeof(G1Jacobian); // buckets estimate
        if (!check_gpu_memory(mem_needed, "MSM benchmark")) {
            printf("  n=2^%d: SKIPPED (insufficient VRAM)\n", log_n);
            continue;
        }

        CudaTimer timer;
        timer.start();
        G1Jacobian result = msm_g1(points.data(), scalars.data(), n);
        float ms = timer.stop();

        printf("  n=2^%d (%d points): %.2f ms\n", log_n, n, ms);
    }
}

static bool should_run_benchmarks(int argc, char** argv) {
    const char* env = std::getenv("ZKML_RUN_BENCHMARKS");
    if (env && std::strcmp(env, "1") == 0) {
        return true;
    }

    for (int i = 1; i < argc; i++) {
        if (std::strcmp(argv[i], "--bench") == 0) {
            return true;
        }
    }

    return false;
}

int main(int argc, char** argv) {
    printf("MSM Test Suite\n");
    printf("==============\n");

    test_msm_small();
    test_msm_medium();
    test_msm_g2();
    if (should_run_benchmarks(argc, argv)) {
        test_msm_benchmark();
    } else {
        printf("\n=== MSM Benchmarks ===\n");
        printf("  SKIPPED (use --bench or ZKML_RUN_BENCHMARKS=1)\n");
    }

    printf("\n==============\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
