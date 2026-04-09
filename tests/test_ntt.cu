#include "ntt/ntt.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

int tests_passed = 0;
int tests_failed = 0;

#define TEST(name) printf("  TEST: %-40s ", name)
#define PASS() do { printf("[PASS]\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("[FAIL] %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if (cond) PASS(); else FAIL(msg); } while(0)

using namespace bn254;

// CPU reference NTT (naive O(n^2) DFT for verification)
void naive_ntt(Fr* out, const Fr* in, int n, Fr omega) {
    for (int i = 0; i < n; i++) {
        Fr sum = Fr::zero();
        Fr omega_pow = Fr::from_uint(1); // omega^0
        Fr omega_i = Fr::from_uint(1);

        // Compute omega^i
        for (int k = 0; k < i; k++) {
            omega_i = omega_i * omega;
        }

        Fr omega_ij = Fr::from_uint(1); // omega^(i*j)
        for (int j = 0; j < n; j++) {
            sum = sum + in[j] * omega_ij;
            omega_ij = omega_ij * omega_i;
        }
        out[i] = sum;
    }
}

void test_ntt_small() {
    printf("\n=== NTT Small Tests ===\n");

    // Test with n = 4 (log_n = 2)
    int log_n = 2;
    int n = 1 << log_n;

    Fr omega = compute_omega(log_n);
    Fr omega_inv = compute_omega_inv(log_n);

    // Verify omega^n = 1
    TEST("omega^n = 1 (n=4)");
    Fr omega_n = omega;
    for (int i = 1; i < n; i++) omega_n = omega_n * omega;
    uint64_t std[4];
    omega_n.to_standard(std);
    CHECK(std[0] == 1 && std[1] == 0 && std[2] == 0 && std[3] == 0,
          "omega^n should be 1");

        TEST("omega * omega_inv = 1 (n=4)");
        Fr omega_prod = omega * omega_inv;
        omega_prod.to_standard(std);
        CHECK(std[0] == 1 && std[1] == 0 && std[2] == 0 && std[3] == 0,
            "omega_inv should be multiplicative inverse of omega");

    // Test NTT on simple input
    TEST("NTT forward (n=4)");
    Fr h_input[4];
    h_input[0] = Fr::from_uint(1);
    h_input[1] = Fr::from_uint(2);
    h_input[2] = Fr::from_uint(3);
    h_input[3] = Fr::from_uint(4);

    // CPU reference
    Fr cpu_output[4];
    naive_ntt(cpu_output, h_input, n, omega);

    // GPU NTT
    Fr *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(Fr)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(Fr)));
    CUDA_CHECK(cudaMemcpy(d_in, h_input, n * sizeof(Fr), cudaMemcpyHostToDevice));

    ntt_gpu(d_out, d_in, log_n, false);
    CUDA_CHECK(cudaDeviceSynchronize());

    Fr gpu_output[4];
    CUDA_CHECK(cudaMemcpy(gpu_output, d_out, n * sizeof(Fr), cudaMemcpyDeviceToHost));

    // Compare (check element 0: sum of inputs = 1+2+3+4 = 10)
    uint64_t gpu_std[4];
    gpu_output[0].to_standard(gpu_std);
    uint64_t cpu_std[4];
    cpu_output[0].to_standard(cpu_std);
    CHECK(gpu_std[0] == cpu_std[0], "NTT[0] should match CPU reference");

    // Test round-trip: iNTT(NTT(x)) = x
    TEST("NTT round-trip (n=4)");
    Fr *d_recovered;
    CUDA_CHECK(cudaMalloc(&d_recovered, n * sizeof(Fr)));
    ntt_gpu(d_recovered, d_out, log_n, true); // inverse NTT
    CUDA_CHECK(cudaDeviceSynchronize());

    Fr recovered[4];
    CUDA_CHECK(cudaMemcpy(recovered, d_recovered, n * sizeof(Fr), cudaMemcpyDeviceToHost));

    bool round_trip_ok = true;
    for (int i = 0; i < n; i++) {
        uint64_t orig[4], rec[4];
        h_input[i].to_standard(orig);
        recovered[i].to_standard(rec);
        if (orig[0] != rec[0]) {
            round_trip_ok = false;
            printf("\n    Mismatch at %d: expected %llu, got %llu",
                   i, (unsigned long long)orig[0], (unsigned long long)rec[0]);
        }
    }
    CHECK(round_trip_ok, "iNTT(NTT(x)) should equal x");

    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_recovered));
}

void test_ntt_medium() {
    printf("\n=== NTT Medium Tests ===\n");

    int log_n = 10;
    int n = 1 << log_n;

    TEST("NTT round-trip (n=1024)");

    std::vector<Fr> h_input(n);
    for (int i = 0; i < n; i++) {
        h_input[i] = Fr::from_uint((uint64_t)(i + 1));
    }

    Fr *d_data, *d_ntt, *d_recovered;
    CUDA_CHECK(cudaMalloc(&d_data, n * sizeof(Fr)));
    CUDA_CHECK(cudaMalloc(&d_ntt, n * sizeof(Fr)));
    CUDA_CHECK(cudaMalloc(&d_recovered, n * sizeof(Fr)));
    CUDA_CHECK(cudaMemcpy(d_data, h_input.data(), n * sizeof(Fr), cudaMemcpyHostToDevice));

    CudaTimer timer;
    timer.start();
    ntt_gpu(d_ntt, d_data, log_n, false);
    CUDA_CHECK(cudaDeviceSynchronize());
    float fwd_ms = timer.stop();

    timer.start();
    ntt_gpu(d_recovered, d_ntt, log_n, true);
    CUDA_CHECK(cudaDeviceSynchronize());
    float inv_ms = timer.stop();

    std::vector<Fr> recovered(n);
    CUDA_CHECK(cudaMemcpy(recovered.data(), d_recovered, n * sizeof(Fr), cudaMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < n; i++) {
        uint64_t orig[4], rec[4];
        h_input[i].to_standard(orig);
        recovered[i].to_standard(rec);
        if (orig[0] != rec[0] || orig[1] != rec[1]) {
            ok = false;
            if (i < 5) printf("\n    Mismatch at %d", i);
        }
    }
    CHECK(ok, "round-trip should be exact");
    printf("    Forward: %.2f ms, Inverse: %.2f ms\n", fwd_ms, inv_ms);

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_ntt));
    CUDA_CHECK(cudaFree(d_recovered));
}

void test_ntt_benchmark() {
    printf("\n=== NTT Benchmarks ===\n");

    for (int log_n = 12; log_n <= 18; log_n += 2) {
        int n = 1 << log_n;
        size_t bytes = (size_t)n * sizeof(Fr);

        if (!check_gpu_memory(bytes * 3, "NTT benchmark")) {
            printf("  n=2^%d: SKIPPED (insufficient VRAM)\n", log_n);
            continue;
        }

        Fr *d_in, *d_out;
        CUDA_CHECK(cudaMalloc(&d_in, bytes));
        CUDA_CHECK(cudaMalloc(&d_out, bytes));

        // Initialize with dummy data
        CUDA_CHECK(cudaMemset(d_in, 0, bytes));

        CudaTimer timer;
        timer.start();
        ntt_gpu(d_out, d_in, log_n, false);
        CUDA_CHECK(cudaDeviceSynchronize());
        float ms = timer.stop();

        printf("  n=2^%d (%d): %.2f ms\n", log_n, n, ms);

        CUDA_CHECK(cudaFree(d_in));
        CUDA_CHECK(cudaFree(d_out));
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
    printf("NTT Test Suite\n");
    printf("==============\n");

    test_ntt_small();
    test_ntt_medium();
    if (should_run_benchmarks(argc, argv)) {
        test_ntt_benchmark();
    } else {
        printf("\n=== NTT Benchmarks ===\n");
        printf("  SKIPPED (use --bench or ZKML_RUN_BENCHMARKS=1)\n");
    }

    printf("\n==============\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    return tests_failed > 0 ? 1 : 0;
}
