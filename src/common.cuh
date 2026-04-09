#pragma once

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cuda_runtime.h>

// ============================================================
// CUDA error checking macro — used on EVERY CUDA API call
// ============================================================
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d — %s\n",                      \
                    __FILE__, __LINE__, cudaGetErrorString(err));                \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// ============================================================
// Memory diagnostics for 4GB VRAM budget (GTX 1650)
// ============================================================
inline void print_gpu_memory(const char* label) {
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    double free_mb = free_bytes / (1024.0 * 1024.0);
    double total_mb = total_bytes / (1024.0 * 1024.0);
    double used_mb = total_mb - free_mb;
    printf("[GPU MEM] %s: used=%.1f MB, free=%.1f MB, total=%.1f MB\n",
           label, used_mb, free_mb, total_mb);
}

// Check that we have enough VRAM before a large allocation
inline bool check_gpu_memory(size_t required_bytes, const char* label) {
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (free_bytes < required_bytes) {
        double req_mb = required_bytes / (1024.0 * 1024.0);
        double free_mb = free_bytes / (1024.0 * 1024.0);
        fprintf(stderr, "[GPU MEM] WARNING: %s needs %.1f MB but only %.1f MB free\n",
                label, req_mb, free_mb);
        return false;
    }
    return true;
}

// Safe cudaMalloc with memory check
template<typename T>
inline cudaError_t safe_cuda_malloc(T** ptr, size_t count, const char* label) {
    size_t bytes = count * sizeof(T);
    if (!check_gpu_memory(bytes, label)) {
        *ptr = nullptr;
        return cudaErrorMemoryAllocation;
    }
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "[GPU MEM] cudaMalloc failed for %s: %s\n",
                label, cudaGetErrorString(err));
        *ptr = nullptr;
    }
    return err;
}

// ============================================================
// SM_75 (GTX 1650 Turing) hardware constants
// ============================================================
namespace gpu_config {
    constexpr int WARP_SIZE = 32;
    constexpr int MAX_THREADS_PER_BLOCK = 1024;
    constexpr int MAX_SHARED_MEM_PER_SM = 65536;  // 64KB
    constexpr int NUM_SMS = 16;
    constexpr int MAX_BLOCKS_PER_SM = 16;
    constexpr int MAX_WARPS_PER_SM = 32;
    constexpr size_t VRAM_BUDGET = 3500ULL * 1024 * 1024; // 3.5GB usable
    constexpr size_t MAX_SINGLE_ALLOC = 1024ULL * 1024 * 1024; // 1GB max
    constexpr int MSM_CHUNK_SIZE = 1 << 20; // 1M points per MSM chunk
    constexpr int NTT_MAX_LOG = 24; // max NTT size = 2^24
}

// ============================================================
// 256-bit unsigned integer type (4 x uint64_t, little-endian)
// ============================================================
struct alignas(32) uint256_t {
    uint64_t limbs[4]; // limbs[0] = least significant

    __host__ __device__ uint256_t() : limbs{0, 0, 0, 0} {}

    __host__ __device__ uint256_t(uint64_t v) : limbs{v, 0, 0, 0} {}

    __host__ __device__ uint256_t(uint64_t l0, uint64_t l1, uint64_t l2, uint64_t l3)
        : limbs{l0, l1, l2, l3} {}

    __host__ __device__ bool operator==(const uint256_t& o) const {
        return limbs[0] == o.limbs[0] && limbs[1] == o.limbs[1] &&
               limbs[2] == o.limbs[2] && limbs[3] == o.limbs[3];
    }

    __host__ __device__ bool operator!=(const uint256_t& o) const {
        return !(*this == o);
    }

    // Compare: returns -1, 0, or 1
    __host__ __device__ int compare(const uint256_t& o) const {
        for (int i = 3; i >= 0; i--) {
            if (limbs[i] < o.limbs[i]) return -1;
            if (limbs[i] > o.limbs[i]) return 1;
        }
        return 0;
    }

    __host__ __device__ bool operator<(const uint256_t& o) const { return compare(o) < 0; }
    __host__ __device__ bool operator>(const uint256_t& o) const { return compare(o) > 0; }
    __host__ __device__ bool operator<=(const uint256_t& o) const { return compare(o) <= 0; }
    __host__ __device__ bool operator>=(const uint256_t& o) const { return compare(o) >= 0; }

    __host__ __device__ bool is_zero() const {
        return limbs[0] == 0 && limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0;
    }

    __host__ __device__ bool bit(int i) const {
        int word = i / 64;
        int bit_pos = i % 64;
        return (limbs[word] >> bit_pos) & 1ULL;
    }

    // Highest set bit position (0-indexed), returns -1 if zero
    __host__ __device__ int highest_bit() const {
        for (int i = 3; i >= 0; i--) {
            if (limbs[i] != 0) {
                // Count leading zeros
                int clz = 0;
                uint64_t v = limbs[i];
                #ifdef __CUDA_ARCH__
                clz = __clzll(v);
                #else
                if (v == 0) { clz = 64; }
                else {
                    clz = 0;
                    if ((v & 0xFFFFFFFF00000000ULL) == 0) { clz += 32; v <<= 32; }
                    if ((v & 0xFFFF000000000000ULL) == 0) { clz += 16; v <<= 16; }
                    if ((v & 0xFF00000000000000ULL) == 0) { clz += 8;  v <<= 8;  }
                    if ((v & 0xF000000000000000ULL) == 0) { clz += 4;  v <<= 4;  }
                    if ((v & 0xC000000000000000ULL) == 0) { clz += 2;  v <<= 2;  }
                    if ((v & 0x8000000000000000ULL) == 0) { clz += 1; }
                }
                #endif
                return i * 64 + (63 - clz);
            }
        }
        return -1;
    }

    // Add with carry, returns carry
    __host__ __device__ static uint256_t add_with_carry(const uint256_t& a,
                                                         const uint256_t& b,
                                                         uint64_t& carry_out) {
        uint256_t result;
        uint64_t carry = 0;
        // Portable add with carry (no __int128 needed)
        for (int i = 0; i < 4; i++) {
            uint64_t sum = a.limbs[i] + b.limbs[i];
            uint64_t c1 = (sum < a.limbs[i]) ? 1ULL : 0ULL;
            uint64_t sum2 = sum + carry;
            uint64_t c2 = (sum2 < sum) ? 1ULL : 0ULL;
            result.limbs[i] = sum2;
            carry = c1 + c2;
        }
        carry_out = carry;
        return result;
    }

    // Subtract with borrow, returns borrow
    __host__ __device__ static uint256_t sub_with_borrow(const uint256_t& a,
                                                          const uint256_t& b,
                                                          uint64_t& borrow_out) {
        uint256_t result;
        uint64_t borrow = 0;
        for (int i = 0; i < 4; i++) {
            uint64_t diff = a.limbs[i] - b.limbs[i];
            uint64_t b1 = (diff > a.limbs[i]) ? 1ULL : 0ULL;
            uint64_t diff2 = diff - borrow;
            uint64_t b2 = (diff2 > diff) ? 1ULL : 0ULL;
            result.limbs[i] = diff2;
            borrow = b1 + b2;
        }
        borrow_out = borrow;
        return result;
    }

    void print(const char* label = "") const {
        printf("%s0x%016llx%016llx%016llx%016llx\n", label,
               (unsigned long long)limbs[3], (unsigned long long)limbs[2],
               (unsigned long long)limbs[1], (unsigned long long)limbs[0]);
    }
};

// ============================================================
// Timer utility for benchmarks
// ============================================================
struct CudaTimer {
    cudaEvent_t start_event, stop_event;

    CudaTimer() {
        CUDA_CHECK(cudaEventCreate(&start_event));
        CUDA_CHECK(cudaEventCreate(&stop_event));
    }

    ~CudaTimer() {
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    void start() {
        CUDA_CHECK(cudaEventRecord(start_event));
    }

    float stop() {
        CUDA_CHECK(cudaEventRecord(stop_event));
        CUDA_CHECK(cudaEventSynchronize(stop_event));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_event, stop_event));
        return ms;
    }
};
