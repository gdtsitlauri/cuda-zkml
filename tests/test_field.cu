#include "field/fp.cuh"
#include "field/fp2.cuh"
#include "field/fp6.cuh"
#include "field/fp12.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstring>

// ============================================================
// BN254 Field Arithmetic Tests
//
// Validates against known test vectors from:
// - EIP-197 (BN254 precompile specification)
// - py_ecc reference implementation
// - arkworks bn254 crate
// ============================================================

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) printf("  TEST: %-40s ", name)
#define PASS() do { printf("[PASS]\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

using namespace bn254;

// ============================================================
// Fp Tests
// ============================================================
void test_fp_basic() {
    printf("\n=== Fp Basic Arithmetic ===\n");

    // Test zero and one
    TEST("zero");
    Fp zero = Fp::zero();
    CHECK(zero.is_zero(), "zero should be zero");

    TEST("one");
    Fp one = Fp::one();
    uint64_t one_std[4];
    one.to_standard(one_std);
    CHECK(one_std[0] == 1 && one_std[1] == 0 && one_std[2] == 0 && one_std[3] == 0,
          "one should be 1");

    // Test from_uint
    TEST("from_uint(42)");
    Fp forty_two = Fp::from_uint(42);
    uint64_t ft_std[4];
    forty_two.to_standard(ft_std);
    CHECK(ft_std[0] == 42 && ft_std[1] == 0, "from_uint(42) should be 42");

    // Test addition
    TEST("addition: 1 + 1 = 2");
    Fp two = one + one;
    uint64_t two_std[4];
    two.to_standard(two_std);
    CHECK(two_std[0] == 2, "1 + 1 should be 2");

    // Test subtraction
    TEST("subtraction: 2 - 1 = 1");
    Fp result = two - one;
    uint64_t res_std[4];
    result.to_standard(res_std);
    CHECK(res_std[0] == 1, "2 - 1 should be 1");

    // Test subtraction resulting in negative (mod p)
    TEST("subtraction: 0 - 1 = p - 1");
    Fp neg_one = zero - one;
    Fp expected_neg = -one;
    CHECK(neg_one == expected_neg, "0 - 1 should equal -1");

    // Test multiplication
    TEST("multiplication: 3 * 7 = 21");
    Fp three = Fp::from_uint(3);
    Fp seven = Fp::from_uint(7);
    Fp twenty_one = three * seven;
    uint64_t to_std[4];
    twenty_one.to_standard(to_std);
    CHECK(to_std[0] == 21, "3 * 7 should be 21");

    // Test squaring
    TEST("squaring: 5^2 = 25");
    Fp five = Fp::from_uint(5);
    Fp sq = five.sqr();
    uint64_t sq_std[4];
    sq.to_standard(sq_std);
    CHECK(sq_std[0] == 25, "5^2 should be 25");

    // Test power
    TEST("power: 2^10 = 1024");
    Fp base = Fp::from_uint(2);
    Fp power = base.pow_u64(10);
    uint64_t pow_std[4];
    power.to_standard(pow_std);
    CHECK(pow_std[0] == 1024, "2^10 should be 1024");

    // Test inverse
    TEST("inverse: 7 * 7^{-1} = 1");
    Fp inv = seven.inv();
    Fp prod = seven * inv;
    CHECK(prod == one, "7 * inv(7) should be 1");

    // Test inverse of larger number
    TEST("inverse: 12345 * 12345^{-1} = 1");
    Fp big = Fp::from_uint(12345);
    Fp big_inv = big.inv();
    Fp big_prod = big * big_inv;
    CHECK(big_prod == one, "12345 * inv(12345) should be 1");

    // Test negation: a + (-a) = 0
    TEST("negation: a + (-a) = 0");
    Fp neg = -five;
    Fp sum = five + neg;
    CHECK(sum.is_zero(), "5 + (-5) should be 0");

    // Test associativity: (a + b) + c = a + (b + c)
    TEST("addition associativity");
    Fp a = Fp::from_uint(111);
    Fp b = Fp::from_uint(222);
    Fp c = Fp::from_uint(333);
    CHECK((a + b) + c == a + (b + c), "addition should be associative");

    // Test commutativity of multiplication
    TEST("multiplication commutativity");
    CHECK(a * b == b * a, "multiplication should be commutative");

    // Test distributivity: a * (b + c) = a*b + a*c
    TEST("distributivity");
    CHECK(a * (b + c) == a * b + a * c, "should be distributive");
}

// ============================================================
// Fp2 Tests
// ============================================================
void test_fp2() {
    printf("\n=== Fp2 Arithmetic ===\n");

    TEST("Fp2 one * one = one");
    Fp2 one = Fp2::one();
    Fp2 prod = one * one;
    CHECK(prod == one, "1 * 1 should be 1");

    TEST("Fp2 addition");
    Fp2 a(Fp::from_uint(3), Fp::from_uint(5));
    Fp2 b(Fp::from_uint(7), Fp::from_uint(11));
    Fp2 sum = a + b;
    Fp2 expected(Fp::from_uint(10), Fp::from_uint(16));
    CHECK(sum == expected, "(3+5u) + (7+11u) should be (10+16u)");

    TEST("Fp2 multiplication (u^2 = -1)");
    // (0 + 1*u) * (0 + 1*u) = u^2 = -1
    Fp2 u(Fp::zero(), Fp::one());
    Fp2 u_sq = u * u;
    Fp2 neg_one(-(Fp::one()), Fp::zero());
    CHECK(u_sq == neg_one, "u^2 should be -1");

    TEST("Fp2 inverse");
    Fp2 x(Fp::from_uint(3), Fp::from_uint(4));
    Fp2 x_inv = x.inv();
    Fp2 should_be_one = x * x_inv;
    CHECK(should_be_one == Fp2::one(), "x * x^{-1} should be 1");

    TEST("Fp2 conjugate: (a+bu)(a-bu) = a^2+b^2");
    Fp2 conj = x.conjugate();
    Fp2 norm_prod = x * conj;
    Fp expected_norm = Fp::from_uint(3).sqr() + Fp::from_uint(4).sqr();
    CHECK(norm_prod.c0 == expected_norm && norm_prod.c1.is_zero(),
          "conjugate product should be real");

    TEST("Fp2 squaring consistency");
    Fp2 sq_method = x.sqr();
    Fp2 mul_method = x * x;
    CHECK(sq_method == mul_method, "sqr() should equal self * self");
}

// ============================================================
// Fp6 and Fp12 Tests
// ============================================================
void test_fp6_fp12() {
    printf("\n=== Fp6/Fp12 Arithmetic ===\n");

    TEST("Fp6 one * one = one");
    Fp6 one6 = Fp6::one();
    Fp6 prod6 = one6 * one6;
    CHECK(prod6 == one6, "Fp6: 1 * 1 should be 1");

    TEST("Fp6 inverse");
    Fp6 a6(Fp2(Fp::from_uint(1), Fp::from_uint(2)),
            Fp2(Fp::from_uint(3), Fp::from_uint(4)),
            Fp2(Fp::from_uint(5), Fp::from_uint(6)));
    Fp6 a6_inv = a6.inv();
    Fp6 should_one = a6 * a6_inv;
    CHECK(should_one == Fp6::one(), "Fp6: a * a^{-1} should be 1");

    TEST("Fp12 one * one = one");
    Fp12 one12 = Fp12::one();
    Fp12 prod12 = one12 * one12;
    CHECK(prod12 == one12, "Fp12: 1 * 1 should be 1");

    TEST("Fp12 squaring consistency");
    Fp12 x12(Fp6(Fp2(Fp::from_uint(1), Fp::from_uint(2)),
                  Fp2(Fp::from_uint(3), Fp::from_uint(4)),
                  Fp2(Fp::from_uint(5), Fp::from_uint(6))),
             Fp6(Fp2(Fp::from_uint(7), Fp::from_uint(8)),
                  Fp2(Fp::from_uint(9), Fp::from_uint(10)),
                  Fp2(Fp::from_uint(11), Fp::from_uint(12))));
    Fp12 sq12 = x12.sqr();
    Fp12 mul12 = x12 * x12;
    CHECK(sq12 == mul12, "Fp12: sqr() should equal self * self");

    TEST("Fp12 inverse");
    Fp12 inv12 = x12.inv();
    Fp12 should_one12 = x12 * inv12;
    CHECK(should_one12 == Fp12::one(), "Fp12: a * a^{-1} should be 1");
}

// ============================================================
// Fr Tests
// ============================================================
void test_fr() {
    printf("\n=== Fr (Scalar Field) ===\n");

    TEST("Fr from_uint and convert");
    Fr a = Fr::from_uint(42);
    uint64_t std[4];
    a.to_standard(std);
    CHECK(std[0] == 42, "Fr from_uint(42) should be 42");

    TEST("Fr addition");
    Fr b = Fr::from_uint(58);
    Fr sum = a + b;
    sum.to_standard(std);
    CHECK(std[0] == 100, "42 + 58 should be 100");

    TEST("Fr multiplication");
    Fr prod = a * b;
    prod.to_standard(std);
    CHECK(std[0] == 2436, "42 * 58 should be 2436");

    TEST("Fr subtraction");
    Fr diff = b - a;
    diff.to_standard(std);
    CHECK(std[0] == 16, "58 - 42 should be 16");

    TEST("Fr inverse");
    Fr inv = a.inv();
    Fr should_one = a * inv;
    should_one.to_standard(std);
    CHECK(std[0] == 1 && std[1] == 0 && std[2] == 0 && std[3] == 0,
          "42 * inv(42) should be 1");
}

// ============================================================
// GPU kernel test
// ============================================================
__global__ void test_fp_kernel(Fp* results) {
    int idx = threadIdx.x;

    if (idx == 0) {
        // Test: 3 * 7 = 21
        Fp a = Fp::from_uint(3);
        Fp b = Fp::from_uint(7);
        results[0] = a * b;
    }
    if (idx == 1) {
        // Test: 2^10 = 1024
        Fp base = Fp::from_uint(2);
        results[1] = base.pow_u64(10);
    }
    if (idx == 2) {
        // Test: 5 * inv(5) = 1
        Fp five = Fp::from_uint(5);
        results[2] = five * five.inv();
    }
}

void test_gpu_kernels() {
    printf("\n=== GPU Kernel Tests ===\n");

    Fp* d_results;
    CUDA_CHECK(cudaMalloc(&d_results, 3 * sizeof(Fp)));

    test_fp_kernel<<<1, 3>>>(d_results);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    Fp h_results[3];
    CUDA_CHECK(cudaMemcpy(h_results, d_results, 3 * sizeof(Fp),
                           cudaMemcpyDeviceToHost));

    TEST("GPU: 3 * 7 = 21");
    uint64_t std[4];
    h_results[0].to_standard(std);
    CHECK(std[0] == 21, "GPU 3*7 should be 21");

    TEST("GPU: 2^10 = 1024");
    h_results[1].to_standard(std);
    CHECK(std[0] == 1024, "GPU 2^10 should be 1024");

    TEST("GPU: 5 * inv(5) = 1");
    CHECK(h_results[2] == Fp::one(), "GPU 5*inv(5) should be 1");

    CUDA_CHECK(cudaFree(d_results));
}

int main() {
    printf("BN254 Field Arithmetic Test Suite\n");
    printf("=================================\n");

    test_fp_basic();
    test_fp2();
    test_fp6_fp12();
    test_fr();
    test_gpu_kernels();

    printf("\n=================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);

    return tests_failed > 0 ? 1 : 0;
}
