#pragma once

// Host reference implementation of the exact integer layers (roadmap 2.1/2.2).
// Semantics (shared with python/zkml/model.py and the v2 R1CS statement):
//   RELU_EXACT: q = floor(max(z, 0) / 2^shift); z must lie in [-2^(bits-1), 2^(bits-1))
//   CONV2D    : y[oc][oy][ox] = b[oc] + sum_{ic,ky,kx} W[oc][ic][ky][kx] * x[ic][oy*s+ky-p][ox*s+kx-p]
// Values are signed integers embedded in Fp; |value| < 2^62 is required.

#include "nn/inference.cuh"
#include <cstdio>
#include <vector>

namespace zkml {

inline bool fp_to_i64_checked(const bn254::Fp& x, int64_t* out) {
    uint64_t v[4];
    x.to_standard(v);
    if (v[1] == 0 && v[2] == 0 && v[3] == 0 && v[0] < (1ULL << 62)) { *out = (int64_t)v[0]; return true; }
    bn254::Fp n = -x;
    n.to_standard(v);
    if (v[1] == 0 && v[2] == 0 && v[3] == 0 && v[0] < (1ULL << 62)) { *out = -(int64_t)v[0]; return true; }
    return false;
}

inline bn254::Fp i64_to_fp(int64_t v) {
    if (v >= 0) return bn254::Fp::from_uint((uint64_t)v);
    return -bn254::Fp::from_uint((uint64_t)(-(v + 1)) + 1u);
}

inline std::vector<bn254::Fp> run_relu_exact_layer_host(const NNModel::Layer& layer,
                                                        const std::vector<bn254::Fp>& in) {
    std::vector<bn254::Fp> out((size_t)layer.out_size, bn254::Fp::zero());
    const int64_t lo = -(1LL << (layer.relu_exact.bits - 1)), hi = (1LL << (layer.relu_exact.bits - 1));
    for (int i = 0; i < layer.in_size; i++) {
        int64_t z = 0;
        if (!fp_to_i64_checked(in[(size_t)i], &z) || z < lo || z >= hi) {
            fprintf(stderr, "[RELU_EXACT] value %d out of the %d-bit range\n", i, layer.relu_exact.bits);
            z = 0;
        }
        int64_t y = z > 0 ? z : 0;
        out[(size_t)i] = i64_to_fp(y >> layer.relu_exact.shift);
    }
    return out;
}

inline std::vector<bn254::Fp> run_conv2d_layer_host(const NNModel::Layer& layer,
                                                    const std::vector<bn254::Fp>& in) {
    const Conv2DParams& c = layer.conv;
    const int oh = c.out_h(), ow = c.out_w();
    std::vector<bn254::Fp> out((size_t)c.out_size(), bn254::Fp::zero());
    for (int oc = 0; oc < c.out_c; oc++)
        for (int oy = 0; oy < oh; oy++)
            for (int ox = 0; ox < ow; ox++) {
                bn254::Fp acc = layer.bias[(size_t)oc];
                for (int ic = 0; ic < c.in_c; ic++)
                    for (int ky = 0; ky < c.kh; ky++)
                        for (int kx = 0; kx < c.kw; kx++) {
                            int iy = oy * c.stride + ky - c.pad, ix = ox * c.stride + kx - c.pad;
                            if (iy < 0 || ix < 0 || iy >= c.in_h || ix >= c.in_w) continue;
                            acc = acc + layer.weights[(((size_t)oc * c.in_c + ic) * c.kh + ky) * c.kw + kx] *
                                        in[((size_t)ic * c.in_h + iy) * c.in_w + ix];
                        }
                out[((size_t)oc * oh + oy) * ow + ox] = acc;
            }
    return out;
}

}  // namespace zkml
