#pragma once

// ============================================================
// R1CS gadgets for the PCANI v2 statement (roadmap 1.1, 2.1-2.4).
//
// Every gadget works on a TracedBuilder: it adds the constraints AND assigns
// the witness values in the same pass, so the witness cannot drift from the
// circuit (the v1 path keeps separate circuit/witness code for reproducibility
// of the published results). All code is host code: it is unit-tested without
// a GPU in tests/host and used by zkml-prove --statement-v2.
//
// Integer convention: a signed integer v is represented by the field element
// v mod r. Range-checked gadgets take an explicit bit width k and are sound
// only for values in the stated range (the range proof enforces it).
// ============================================================

#include "prover/circuit.cuh"
#include "prover/poseidon_constants.cuh"
#include "nn/layers.cuh"
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace zkml {

using bn254::Fr;
using LC = std::vector<std::pair<int, Fr>>;

// ---------------- field helpers (host) ----------------
inline Fr fr_from_limbs(const uint64_t limbs[4]) {
    // limbs: canonical little-endian value, must be < r
    Fr out;
    uint64_t tmp[4] = {limbs[0], limbs[1], limbs[2], limbs[3]};
    Fr::mont_mul_fr(out.val, tmp, Fr::r_squared().val);
    return out;
}

inline Fr fr_from_i64(int64_t v) {
    if (v >= 0) return Fr::from_uint((uint64_t)v);
    return -Fr::from_uint((uint64_t)(-(v + 1)) + 1u);
}

inline Fr fr_pow2(int k) {
    Fr x = Fr::one();
    Fr two = Fr::from_uint(2);
    for (int i = 0; i < k; i++) x = x * two;
    return x;
}

// Signed view of a field element: true and *out set if |v| < 2^62.
inline bool fr_to_i64(const Fr& x, int64_t* out) {
    uint64_t v[4];
    x.to_standard(v);
    if (v[1] == 0 && v[2] == 0 && v[3] == 0 && v[0] < (1ULL << 62)) {
        *out = (int64_t)v[0];
        return true;
    }
    Fr neg = -x;
    neg.to_standard(v);
    if (v[1] == 0 && v[2] == 0 && v[3] == 0 && v[0] < (1ULL << 62)) {
        *out = -(int64_t)v[0];
        return true;
    }
    return false;
}

// Parse a hex string (optional 0x) into a field element, keeping the low 253 bits
// so the value is always < r.
inline bool fr_from_hex(const std::string& s, Fr* out) {
    std::string h = s;
    if (h.size() >= 2 && h[0] == '0' && (h[1] == 'x' || h[1] == 'X')) h = h.substr(2);
    if (h.empty() || h.size() > 64) return false;
    uint64_t limbs[4] = {0, 0, 0, 0};
    for (char c : h) {
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (c >= 'a' && c <= 'f') d = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') d = c - 'A' + 10;
        else return false;
        // limbs <<= 4
        limbs[3] = (limbs[3] << 4) | (limbs[2] >> 60);
        limbs[2] = (limbs[2] << 4) | (limbs[1] >> 60);
        limbs[1] = (limbs[1] << 4) | (limbs[0] >> 60);
        limbs[0] = (limbs[0] << 4) | (uint64_t)d;
    }
    limbs[3] &= 0x1FFFFFFFFFFFFFFFULL;  // keep 253 bits
    *out = fr_from_limbs(limbs);
    return true;
}

inline std::string fr_to_hex(const Fr& x) {
    uint64_t v[4];
    x.to_standard(v);
    char buf[67];
    snprintf(buf, sizeof(buf), "0x%016llx%016llx%016llx%016llx",
             (unsigned long long)v[3], (unsigned long long)v[2],
             (unsigned long long)v[1], (unsigned long long)v[0]);
    return buf;
}

// Merge repeated variables of a linear combination (keeps R1CS rows canonical).
inline LC lc_compress(const LC& lc) {
    LC out;
    for (auto& [v, c] : lc) {
        bool merged = false;
        for (auto& [ov, oc] : out) {
            if (ov == v) { oc = oc + c; merged = true; break; }
        }
        if (!merged) out.push_back({v, c});
    }
    LC nz;
    for (auto& t : out) if (!t.second.is_zero()) nz.push_back(t);
    return nz;
}

// ---------------- traced builder ----------------
struct TracedBuilder {
    CircuitBuilder cb;
    std::vector<Fr> w;          // w[0] = 1

    TracedBuilder() { w.push_back(Fr::one()); }

    int var(const Fr& value) {
        int idx = cb.alloc_var();
        if ((int)w.size() <= idx) w.resize((size_t)idx + 1, Fr::zero());
        w[(size_t)idx] = value;
        return idx;
    }
    int var_unset() { return var(Fr::zero()); }
    void set(int idx, const Fr& value) {
        if ((int)w.size() <= idx) w.resize((size_t)idx + 1, Fr::zero());
        w[(size_t)idx] = value;
    }
    Fr val(int idx) const { return w[(size_t)idx]; }
    Fr eval(const LC& lc) const {
        Fr acc = Fr::zero();
        for (auto& [v, c] : lc) acc = acc + c * w[(size_t)v];
        return acc;
    }
    void constrain(const LC& a, const LC& b, const LC& c) {
        cb.circuit.add_constraint(lc_compress(a), lc_compress(b), lc_compress(c));
    }
    void linear(const LC& lhs, int out) { cb.circuit.add_linear_constraint(lc_compress(lhs), {{out, Fr::one()}}); }
    // lc == 0
    void constrain_zero(const LC& lc) { constrain(lc, {{0, Fr::one()}}, {}); }
    // x * y = z for single variables
    void mul(int x, int y, int z) { cb.circuit.add_mul_constraint(x, Fr::one(), y, Fr::one(), z, Fr::one()); }
    R1CS& r1cs() { return cb.circuit; }
};

// ---------------- 1.1 context binding ----------------
// A public input that appears in no constraint has a zero IC term in Groth16
// and could be changed without invalidating the proof. ctx * ctx = ctx_sq
// makes the context a genuine part of the statement (Tornado-style binding).
inline int add_context_binding(TracedBuilder& tb, int ctx_var) {
    Fr c = tb.val(ctx_var);
    int sq = tb.var(c * c);
    tb.mul(ctx_var, ctx_var, sq);
    return sq;
}

// ---------------- range proofs / bit decomposition ----------------
// Decompose (lc + offset) into k boolean variables (LSB first) and constrain
// sum 2^i b_i == lc + offset. Sound iff the value lies in [0, 2^k) with k < 253.
inline std::vector<int> add_bits(TracedBuilder& tb, const LC& lc, const Fr& offset, int k) {
    Fr v = tb.eval(lc) + offset;
    uint64_t limbs[4];
    v.to_standard(limbs);
    std::vector<int> bits((size_t)k);
    LC sum;
    Fr pw = Fr::one();
    Fr two = Fr::from_uint(2);
    for (int i = 0; i < k; i++) {
        uint64_t b = (limbs[i / 64] >> (i % 64)) & 1ULL;
        int bv = tb.var(Fr::from_uint(b));
        bits[(size_t)i] = bv;
        tb.constrain({{bv, Fr::one()}}, {{bv, Fr::one()}, {0, -Fr::one()}}, {});   // b(b-1)=0
        sum.push_back({bv, pw});
        pw = pw * two;
    }
    // sum - lc - offset == 0
    LC diff = sum;
    for (auto& [var, c] : lc) diff.push_back({var, -c});
    diff.push_back({0, -offset});
    tb.constrain_zero(diff);
    return bits;
}

// Range check: x in [-2^(k-1), 2^(k-1)).
inline std::vector<int> add_signed_range(TracedBuilder& tb, int x, int k) {
    return add_bits(tb, {{x, Fr::one()}}, fr_pow2(k - 1), k);
}

// ---------------- 2.1 exact ReLU ----------------
// For z in [-2^(k-1), 2^(k-1)): s = msb(z + 2^(k-1)) = [z >= 0], y = s * z.
inline int add_relu_exact(TracedBuilder& tb, int z, int k) {
    std::vector<int> bits = add_signed_range(tb, z, k);
    int s = bits[(size_t)k - 1];
    int y = tb.var(tb.val(s) * tb.val(z));
    tb.mul(s, z, y);
    return y;
}

// Floor division by 2^shift with range-checked remainder and quotient:
// y = q * 2^shift + r, 0 <= r < 2^shift, q in [-2^(kq-1), 2^(kq-1)).
inline int add_rescale(TracedBuilder& tb, int y, int shift, int kq) {
    if (shift <= 0) return y;
    int64_t yv = 0;
    fr_to_i64(tb.val(y), &yv);
    int64_t q = yv >> shift;                                  // arithmetic shift = floor
    int64_t r = yv - (q << shift);
    int qv = tb.var(fr_from_i64(q));
    int rv = tb.var(fr_from_i64(r));
    add_bits(tb, {{rv, Fr::one()}}, Fr::zero(), shift);       // 0 <= r < 2^shift
    add_signed_range(tb, qv, kq);
    tb.constrain_zero({{qv, fr_pow2(shift)}, {rv, Fr::one()}, {y, -Fr::one()}});
    return qv;
}

// Comparison flag: f = [x - y - t >= 0] for (x - y - t) in [-2^(k-1), 2^(k-1)).
// Used for in-circuit routing on private data (roadmap 2.4).
inline int add_geq_flag(TracedBuilder& tb, int x, int y, int64_t t, int k) {
    LC d = {{x, Fr::one()}, {y, -Fr::one()}, {0, -fr_from_i64(t)}};
    std::vector<int> bits = add_bits(tb, d, fr_pow2(k - 1), k);
    return bits[(size_t)k - 1];
}

// ---------------- linear layers ----------------
// Fixed (public) coefficients; zero weights produce no terms (sparse R1CS rows).
inline std::vector<int> add_fixed_linear(TracedBuilder& tb, const std::vector<int>& x,
                                         const std::vector<Fr>& weights, const std::vector<Fr>& bias,
                                         int out, int in, const std::vector<int>* y_preset = nullptr) {
    std::vector<int> y((size_t)out);
    for (int i = 0; i < out; i++) {
        Fr acc = bias[(size_t)i];
        LC terms;
        for (int j = 0; j < in; j++) {
            const Fr& wij = weights[(size_t)i * (size_t)in + (size_t)j];
            if (wij.is_zero()) continue;
            terms.push_back({x[(size_t)j], wij});
            acc = acc + wij * tb.val(x[(size_t)j]);
        }
        terms.push_back({0, bias[(size_t)i]});
        if (y_preset) { y[(size_t)i] = (*y_preset)[(size_t)i]; tb.set(y[(size_t)i], acc); }
        else y[(size_t)i] = tb.var(acc);
        tb.cb.circuit.add_linear_constraint(terms, {{y[(size_t)i], Fr::one()}});
    }
    return y;
}

// Private weights (roadmap 2.3): each weight x input product is a real constraint.
inline std::vector<int> add_private_linear(TracedBuilder& tb, const std::vector<int>& x,
                                           const std::vector<int>& w_vars, const std::vector<int>& b_vars,
                                           int out, int in, const std::vector<int>* y_preset = nullptr) {
    std::vector<int> y((size_t)out);
    for (int i = 0; i < out; i++) {
        LC terms;
        Fr acc = tb.val(b_vars[(size_t)i]);
        for (int j = 0; j < in; j++) {
            int wv = w_vars[(size_t)i * (size_t)in + (size_t)j];
            Fr p = tb.val(wv) * tb.val(x[(size_t)j]);
            int pv = tb.var(p);
            tb.mul(wv, x[(size_t)j], pv);
            terms.push_back({pv, Fr::one()});
            acc = acc + p;
        }
        terms.push_back({b_vars[(size_t)i], Fr::one()});
        if (y_preset) { y[(size_t)i] = (*y_preset)[(size_t)i]; tb.set(y[(size_t)i], acc); }
        else y[(size_t)i] = tb.var(acc);
        tb.cb.circuit.add_linear_constraint(terms, {{y[(size_t)i], Fr::one()}});
    }
    return y;
}

// ---------------- 2.2 native convolution ----------------
using Conv2DShape = Conv2DParams;   // nn/layers.cuh

// Layout: x[c][h][w] (CHW, flattened), weights[oc][ic][ky][kx], bias[oc].
// Only in-bounds, non-zero taps generate terms (no dense unrolling).
inline std::vector<int> add_fixed_conv2d(TracedBuilder& tb, const std::vector<int>& x, const Conv2DShape& s,
                                         const std::vector<Fr>& weights, const std::vector<Fr>& bias,
                                         const std::vector<int>* y_preset = nullptr) {
    const int oh = s.out_h(), ow = s.out_w();
    std::vector<int> y((size_t)s.out_size());
    for (int oc = 0; oc < s.out_c; oc++) {
        for (int oy = 0; oy < oh; oy++) {
            for (int ox = 0; ox < ow; ox++) {
                LC terms;
                Fr acc = bias[(size_t)oc];
                for (int ic = 0; ic < s.in_c; ic++) {
                    for (int ky = 0; ky < s.kh; ky++) {
                        for (int kx = 0; kx < s.kw; kx++) {
                            int iy = oy * s.stride + ky - s.pad;
                            int ix = ox * s.stride + kx - s.pad;
                            if (iy < 0 || ix < 0 || iy >= s.in_h || ix >= s.in_w) continue;
                            const Fr& wv = weights[(((size_t)oc * s.in_c + ic) * s.kh + ky) * s.kw + kx];
                            if (wv.is_zero()) continue;
                            int xv = x[((size_t)ic * s.in_h + iy) * s.in_w + ix];
                            terms.push_back({xv, wv});
                            acc = acc + wv * tb.val(xv);
                        }
                    }
                }
                terms.push_back({0, bias[(size_t)oc]});
                size_t o = ((size_t)oc * oh + oy) * ow + ox;
                if (y_preset) { y[o] = (*y_preset)[o]; tb.set(y[o], acc); }
                else y[o] = tb.var(acc);
                tb.cb.circuit.add_linear_constraint(terms, {{y[o], Fr::one()}});
            }
        }
    }
    return y;
}

// Private-weight convolution (roadmap 2.3): one multiplication constraint per tap.
inline std::vector<int> add_private_conv2d(TracedBuilder& tb, const std::vector<int>& x, const Conv2DShape& s,
                                           const std::vector<int>& w_vars, const std::vector<int>& b_vars,
                                           const std::vector<int>* y_preset = nullptr) {
    const int oh = s.out_h(), ow = s.out_w();
    std::vector<int> y((size_t)s.out_size());
    for (int oc = 0; oc < s.out_c; oc++)
        for (int oy = 0; oy < oh; oy++)
            for (int ox = 0; ox < ow; ox++) {
                LC terms;
                Fr acc = tb.val(b_vars[(size_t)oc]);
                for (int ic = 0; ic < s.in_c; ic++)
                    for (int ky = 0; ky < s.kh; ky++)
                        for (int kx = 0; kx < s.kw; kx++) {
                            int iy = oy * s.stride + ky - s.pad, ix = ox * s.stride + kx - s.pad;
                            if (iy < 0 || ix < 0 || iy >= s.in_h || ix >= s.in_w) continue;
                            int wv = w_vars[(((size_t)oc * s.in_c + ic) * s.kh + ky) * s.kw + kx];
                            int xv = x[((size_t)ic * s.in_h + iy) * s.in_w + ix];
                            Fr p = tb.val(wv) * tb.val(xv);
                            int pv = tb.var(p);
                            tb.mul(wv, xv, pv);
                            terms.push_back({pv, Fr::one()});
                            acc = acc + p;
                        }
                terms.push_back({b_vars[(size_t)oc], Fr::one()});
                size_t o = ((size_t)oc * oh + oy) * ow + ox;
                if (y_preset) { y[o] = (*y_preset)[o]; tb.set(y[o], acc); }
                else y[o] = tb.var(acc);
                tb.cb.circuit.add_linear_constraint(terms, {{y[o], Fr::one()}});
            }
    return y;
}

// ---------------- 2.3/2.4 Poseidon (BN254, t=3, circomlib instance) ----------------
struct PoseidonConsts {
    std::vector<Fr> rc;
    Fr mds[3][3];
    PoseidonConsts() {
        using namespace poseidon_bn254;
        const int n = (RF + RP) * T;
        rc.resize((size_t)n);
        for (int i = 0; i < n; i++) rc[(size_t)i] = fr_from_limbs(ROUND_CONSTANTS[i]);
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++) mds[i][j] = fr_from_limbs(MDS[i][j]);
    }
    static const PoseidonConsts& get() {
        static PoseidonConsts c;
        return c;
    }
};

inline void poseidon_permute_native(Fr s[3]) {
    const auto& P = PoseidonConsts::get();
    const int RF = poseidon_bn254::RF, RP = poseidon_bn254::RP, half = RF / 2;
    for (int r = 0; r < RF + RP; r++) {
        for (int i = 0; i < 3; i++) s[i] = s[i] + P.rc[(size_t)r * 3 + i];
        auto sbox = [](const Fr& x) { Fr x2 = x * x; Fr x4 = x2 * x2; return x4 * x; };
        if (r < half || r >= half + RP) { for (int i = 0; i < 3; i++) s[i] = sbox(s[i]); }
        else s[0] = sbox(s[0]);
        Fr n[3];
        for (int i = 0; i < 3; i++) n[i] = P.mds[i][0] * s[0] + P.mds[i][1] * s[1] + P.mds[i][2] * s[2];
        for (int i = 0; i < 3; i++) s[i] = n[i];
    }
}

inline Fr poseidon_hash2_native(const Fr& a, const Fr& b) {
    Fr s[3] = {Fr::zero(), a, b};
    poseidon_permute_native(s);
    return s[0];
}

// In-circuit permutation. The state is carried as linear combinations; only the
// S-boxes cost constraints (3 per S-box: x^2, x^4, x^5), 243 per permutation.
inline void add_poseidon_permute(TracedBuilder& tb, LC st[3]) {
    const auto& P = PoseidonConsts::get();
    const int RF = poseidon_bn254::RF, RP = poseidon_bn254::RP, half = RF / 2;
    auto sbox = [&](const LC& x) -> LC {
        Fr xv = tb.eval(x);
        int x2 = tb.var(xv * xv);
        tb.constrain(x, x, {{x2, Fr::one()}});
        int x4 = tb.var(tb.val(x2) * tb.val(x2));
        tb.mul(x2, x2, x4);
        int x5 = tb.var(tb.val(x4) * xv);
        tb.constrain({{x4, Fr::one()}}, x, {{x5, Fr::one()}});
        return LC{{x5, Fr::one()}};
    };
    for (int r = 0; r < RF + RP; r++) {
        for (int i = 0; i < 3; i++) st[i].push_back({0, P.rc[(size_t)r * 3 + i]});
        if (r < half || r >= half + RP) { for (int i = 0; i < 3; i++) st[i] = sbox(st[i]); }
        else st[0] = sbox(st[0]);
        LC n[3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                for (auto& [v, c] : st[j]) n[i].push_back({v, P.mds[i][j] * c});
        for (int i = 0; i < 3; i++) st[i] = lc_compress(n[i]);
        // keep linear combinations short: materialise the state after each round
        for (int i = 0; i < 3; i++) {
            if (st[i].size() > 8) {
                int m = tb.var(tb.eval(st[i]));
                tb.linear(st[i], m);
                st[i] = LC{{m, Fr::one()}};
            }
        }
    }
}

inline int add_poseidon_hash2(TracedBuilder& tb, const LC& a, const LC& b) {
    LC st[3] = {LC{}, a, b};
    add_poseidon_permute(tb, st);
    int out = tb.var(tb.eval(st[0]));
    tb.linear(st[0], out);
    return out;
}

// Sponge commitment (rate 2, capacity 1; capacity = (len << 64) | domain),
// identical to python/zkml/poseidon.py::commit.
inline Fr poseidon_commit_native(const std::vector<Fr>& vals, uint64_t domain) {
    Fr s[3] = {fr_pow2(64) * Fr::from_uint((uint64_t)vals.size()) + Fr::from_uint(domain), Fr::zero(), Fr::zero()};
    size_t n = vals.empty() ? 2 : vals.size() + (vals.size() % 2);
    for (size_t i = 0; i < n; i += 2) {
        s[1] = s[1] + (i < vals.size() ? vals[i] : Fr::zero());
        s[2] = s[2] + (i + 1 < vals.size() ? vals[i + 1] : Fr::zero());
        poseidon_permute_native(s);
    }
    return s[1];
}

inline int add_poseidon_commit(TracedBuilder& tb, const std::vector<int>& vars, uint64_t domain,
                               int out_preset = -1) {
    LC st[3] = {LC{{0, fr_pow2(64) * Fr::from_uint((uint64_t)vars.size()) + Fr::from_uint(domain)}}, LC{}, LC{}};
    size_t n = vars.empty() ? 2 : vars.size() + (vars.size() % 2);
    for (size_t i = 0; i < n; i += 2) {
        if (i < vars.size()) st[1].push_back({vars[i], Fr::one()});
        if (i + 1 < vars.size()) st[2].push_back({vars[i + 1], Fr::one()});
        add_poseidon_permute(tb, st);
    }
    int out = out_preset;
    if (out < 0) out = tb.var(tb.eval(st[1]));
    else tb.set(out, tb.eval(st[1]));
    tb.linear(st[1], out);
    return out;
}

}  // namespace zkml
