#pragma once

#include "field/fp.cuh"
#include <cmath>

// ============================================================
// Quantization: float32 → integer → Fp field elements
//
// Strategy:
// 1. Symmetric quantization: x_int = round(x * scale)
//    where scale = (2^{bits-1} - 1) / max(|x|)
// 2. Map integers to field elements:
//    - Positive: x_int → Fp::from_uint(x_int)
//    - Negative: x_int → Fp(p + x_int) = -Fp::from_uint(-x_int)
// 3. Accumulation in Fp is exact (no overflow for typical NN sizes)
//
// For MNIST MLP (784→128→10):
//   Max accumulation = 784 * 127 * 127 ≈ 12.6M < p (huge)
//   So quantization to int8 is safe.
//
// For larger networks, use int16 with scale adjustment.
// ============================================================

namespace zkml {

struct QuantConfig {
    int bits;           // Quantization bits (8 or 16)
    float scale;        // Quantization scale factor
    int zero_point;     // Zero point (0 for symmetric)

    static QuantConfig symmetric(int bits, float max_abs) {
        QuantConfig cfg;
        cfg.bits = bits;
        cfg.scale = ((1 << (bits - 1)) - 1) / max_abs;
        cfg.zero_point = 0;
        return cfg;
    }
};

// Quantize a float value to int32
inline int32_t quantize_value(float x, const QuantConfig& cfg) {
    int32_t max_val = (1 << (cfg.bits - 1)) - 1;
    int32_t min_val = -(1 << (cfg.bits - 1));
    int32_t q = (int32_t)roundf(x * cfg.scale);
    if (q > max_val) q = max_val;
    if (q < min_val) q = min_val;
    return q;
}

// Convert quantized int to Fp field element
inline bn254::Fp int_to_fp(int32_t x) {
    if (x >= 0) {
        return bn254::Fp::from_uint((uint64_t)x);
    } else {
        // Negative: represent as p - |x|
        return -bn254::Fp::from_uint((uint64_t)(-x));
    }
}

// Convert Fp back to integer (for comparison/debugging)
// Assumes the value is either small positive or close to p (negative)
inline int32_t fp_to_int(const bn254::Fp& x) {
    uint64_t std[4];
    x.to_standard(std);

    // Check if it's a "negative" number (close to p)
    // p/2 ≈ 0x1832... so check if std[3] >= 0x1832...
    if (std[3] > 0x18324E72E131A029ULL ||
        (std[3] == 0x18324E72E131A029ULL && std[2] > 0)) {
        // It's negative: compute p - x
        bn254::Fp neg = -x;
        uint64_t neg_std[4];
        neg.to_standard(neg_std);
        return -(int32_t)neg_std[0];
    }
    return (int32_t)std[0];
}

// Quantize an entire float array to Fp array
inline void quantize_array(bn254::Fp* out, const float* in, int n,
                            const QuantConfig& cfg) {
    for (int i = 0; i < n; i++) {
        int32_t q = quantize_value(in[i], cfg);
        out[i] = int_to_fp(q);
    }
}

// Dequantize Fp array back to float (for verification)
inline void dequantize_array(float* out, const bn254::Fp* in, int n,
                              const QuantConfig& cfg) {
    for (int i = 0; i < n; i++) {
        int32_t q = fp_to_int(in[i]);
        out[i] = (float)q / cfg.scale;
    }
}

// Find max absolute value in array
inline float find_max_abs(const float* data, int n) {
    float max_val = 0;
    for (int i = 0; i < n; i++) {
        float abs_val = fabsf(data[i]);
        if (abs_val > max_val) max_val = abs_val;
    }
    return max_val > 0 ? max_val : 1.0f;
}

} // namespace zkml
