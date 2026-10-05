#pragma once

// ============================================================
// Groth16 phase-2 MPC ceremony (roadmap 2.5), host code.
//
// Problem: whoever runs Groth16 setup knows delta (and tau, alpha, beta, gamma)
// and could forge proofs. Phase 2 lets N participants re-randomise delta in
// turn; the final key is sound if AT LEAST ONE participant destroyed its secret.
//
// Contribution with secret s (participant i):
//   delta_g1' = s * delta_g1,  delta_g2' = s * delta_g2   (also vk.delta_g2)
//   L_j'      = s^-1 * L_j,    H_j'      = s^-1 * H_j
// Transcript: S1 = s*G1, S2 = s*G2 and a Schnorr proof of knowledge of s
// (K = k*G1, c = Poseidon(S1, K, delta_before, delta_after), z = k + c*s).
//
// Verification (anyone, from the before/after keys and the transcript):
//   z*G1 == K + c*S1                               (participant knows s)
//   e(S1, G2) == e(G1, S2)                          (S1, S2 share s)
//   e(delta_g1', G2) == e(delta_g1, S2)             (delta_g1 multiplied by s)
//   e(delta_g1, delta_g2') == e(delta_g1', delta_g2) (delta_g2 multiplied by s)
//   e(sum rho_j L_j', delta_g2') == e(sum rho_j L_j, delta_g2)   (random rho, batch)
//   same for H; every other key element unchanged; vk.delta_g2 == pk.delta_g2'.
//
// Limitations (stated in docs/THREAT_MODEL.md): phase 1 (powers of tau) is
// still the single-party setup of Groth16Prover::setup; a production ceremony
// should start from a public multi-party powers-of-tau. This tool covers the
// circuit-specific phase 2 only.
// ============================================================

#include "curve/g1.cuh"
#include "curve/g2.cuh"
#include "curve/pairing.cuh"
#include "prover/gadgets.cuh"
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace zkml {
namespace ceremony {

using bn254::Fp;
using bn254::G1Affine;
using bn254::G1Jacobian;
using bn254::G2Affine;
using bn254::G2Jacobian;

struct PK {   // mirror of the ZKMLPK3 file layout (groth16.cu ProvingKey::save)
    int32_t num_constraints = 0, num_variables = 0, num_public = 0;
    G1Affine alpha_g1, beta_g1, delta_g1;
    G2Affine beta_g2, gamma_g2, delta_g2;
    std::vector<G1Affine> tau_g1, A, B1, L, H;
    std::vector<G2Affine> tau_g2, B2;
};

struct VK {   // mirror of verifier.cu save_vk
    int32_t num_public = 0;
    G1Affine alpha_g1;
    G2Affine beta_g2, gamma_g2, delta_g2;
    std::vector<G1Affine> ic;
};

struct Transcript {
    G1Affine S1, K;
    G2Affine S2;
    Fr z;
};

// ---------------- (de)serialisation ----------------
namespace io {
struct Reader {
    FILE* f; bool ok = true;
    void raw(void* p, size_t n) { if (ok) ok = fread(p, 1, n, f) == n; }
    int32_t i32() { int32_t v = 0; raw(&v, 4); return v; }
    uint32_t u32() { uint32_t v = 0; raw(&v, 4); return v; }
    Fp fp() { uint64_t l[4] = {0, 0, 0, 0}; raw(l, 32); return Fp::from_standard(l[0], l[1], l[2], l[3]); }
    G1Affine g1(bool flag) { uint8_t inf = 0; if (flag) raw(&inf, 1); Fp x = fp(), y = fp(); return inf ? G1Affine() : G1Affine(x, y); }
    G2Affine g2(bool flag) {
        uint8_t inf = 0; if (flag) raw(&inf, 1);
        Fp a = fp(), b = fp(), c = fp(), d = fp();
        return inf ? G2Affine() : G2Affine(bn254::Fp2(a, b), bn254::Fp2(c, d));
    }
    template <class T, class F> void vec(std::vector<T>& v, F one) {
        int32_t n = i32();
        if (!ok || n < 0 || n > (1 << 28)) { ok = false; return; }
        v.resize((size_t)n);
        for (auto& x : v) x = one();
    }
};
struct Writer {
    FILE* f; bool ok = true;
    void raw(const void* p, size_t n) { if (ok) ok = fwrite(p, 1, n, f) == n; }
    void i32(int32_t v) { raw(&v, 4); }
    void u32(uint32_t v) { raw(&v, 4); }
    void fp(const Fp& v) { uint64_t l[4]; v.to_standard(l); raw(l, 32); }
    void g1(const G1Affine& p, bool flag) { if (flag) { uint8_t i = p.infinity ? 1 : 0; raw(&i, 1); } fp(p.x); fp(p.y); }
    void g2(const G2Affine& p, bool flag) {
        if (flag) { uint8_t i = p.infinity ? 1 : 0; raw(&i, 1); }
        fp(p.x.c0); fp(p.x.c1); fp(p.y.c0); fp(p.y.c1);
    }
};
}  // namespace io

inline bool load_pk(const char* path, PK& pk) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    io::Reader r{f};
    char magic[8] = {0};
    r.raw(magic, 8);
    if (!r.ok || std::memcmp(magic, "ZKMLPK3", 7) != 0 || r.u32() != 3) { fclose(f); return false; }
    pk.num_constraints = r.i32(); pk.num_variables = r.i32(); pk.num_public = r.i32();
    pk.alpha_g1 = r.g1(true); pk.beta_g1 = r.g1(true); pk.delta_g1 = r.g1(true);
    pk.beta_g2 = r.g2(true); pk.gamma_g2 = r.g2(true); pk.delta_g2 = r.g2(true);
    r.vec(pk.tau_g1, [&] { return r.g1(true); });
    r.vec(pk.tau_g2, [&] { return r.g2(true); });
    r.vec(pk.A, [&] { return r.g1(true); });
    r.vec(pk.B1, [&] { return r.g1(true); });
    r.vec(pk.B2, [&] { return r.g2(true); });
    r.vec(pk.L, [&] { return r.g1(true); });
    r.vec(pk.H, [&] { return r.g1(true); });
    fclose(f);
    return r.ok;
}

inline bool save_pk(const char* path, const PK& pk) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    io::Writer w{f};
    const char magic[8] = {'Z', 'K', 'M', 'L', 'P', 'K', '3', '\0'};
    w.raw(magic, 8); w.u32(3);
    w.i32(pk.num_constraints); w.i32(pk.num_variables); w.i32(pk.num_public);
    w.g1(pk.alpha_g1, true); w.g1(pk.beta_g1, true); w.g1(pk.delta_g1, true);
    w.g2(pk.beta_g2, true); w.g2(pk.gamma_g2, true); w.g2(pk.delta_g2, true);
    auto v1 = [&](const std::vector<G1Affine>& v) { w.i32((int32_t)v.size()); for (auto& p : v) w.g1(p, true); };
    auto v2 = [&](const std::vector<G2Affine>& v) { w.i32((int32_t)v.size()); for (auto& p : v) w.g2(p, true); };
    v1(pk.tau_g1); v2(pk.tau_g2); v1(pk.A); v1(pk.B1); v2(pk.B2); v1(pk.L); v1(pk.H);
    fclose(f);
    return w.ok;
}

inline bool load_vk(const char* path, VK& vk) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    io::Reader r{f};
    vk.num_public = r.i32();
    vk.alpha_g1 = r.g1(false); vk.beta_g2 = r.g2(false); vk.gamma_g2 = r.g2(false); vk.delta_g2 = r.g2(false);
    r.vec(vk.ic, [&] { return r.g1(false); });
    fclose(f);
    return r.ok;
}

inline bool save_vk(const char* path, const VK& vk) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    io::Writer w{f};
    w.i32(vk.num_public);
    w.g1(vk.alpha_g1, false); w.g2(vk.beta_g2, false); w.g2(vk.gamma_g2, false); w.g2(vk.delta_g2, false);
    w.i32((int32_t)vk.ic.size());
    for (auto& p : vk.ic) w.g1(p, false);
    fclose(f);
    return w.ok;
}

inline bool save_transcript(const char* path, const Transcript& t) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    io::Writer w{f};
    w.raw("ZKMLMPC1", 8);
    w.g1(t.S1, true); w.g1(t.K, true); w.g2(t.S2, true);
    uint64_t z[4]; t.z.to_standard(z); w.raw(z, 32);
    fclose(f);
    return w.ok;
}

inline bool load_transcript(const char* path, Transcript& t) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    io::Reader r{f};
    char magic[8] = {0};
    r.raw(magic, 8);
    if (std::memcmp(magic, "ZKMLMPC1", 8) != 0) { fclose(f); return false; }
    t.S1 = r.g1(true); t.K = r.g1(true); t.S2 = r.g2(true);
    uint64_t z[4] = {0, 0, 0, 0}; r.raw(z, 32);
    t.z = fr_from_limbs(z);
    fclose(f);
    return r.ok;
}

// ---------------- helpers ----------------
inline G1Affine mul1(const G1Affine& p, const Fr& s) { return G1Jacobian::from_affine(p).scalar_mul(s).to_affine(); }
inline G2Affine mul2(const G2Affine& p, const Fr& s) { return G2Jacobian::from_affine(p).scalar_mul(s).to_affine(); }
inline bool eq1(const G1Affine& a, const G1Affine& b) {
    if (a.infinity || b.infinity) return a.infinity == b.infinity;
    return a.x == b.x && a.y == b.y;
}
inline bool eq2(const G2Affine& a, const G2Affine& b) {
    if (a.infinity || b.infinity) return a.infinity == b.infinity;
    return a.x.c0 == b.x.c0 && a.x.c1 == b.x.c1 && a.y.c0 == b.y.c0 && a.y.c1 == b.y.c1;
}

// Fiat-Shamir challenge: Poseidon commitment over the 64-bit limbs of the points.
inline Fr challenge(const G1Affine& S1, const G1Affine& K, const G1Affine& d_before, const G1Affine& d_after) {
    std::vector<Fr> v;
    for (const G1Affine* p : {&S1, &K, &d_before, &d_after}) {
        uint64_t l[4];
        p->x.to_standard(l); for (int i = 0; i < 4; i++) v.push_back(Fr::from_uint(l[i]));
        p->y.to_standard(l); for (int i = 0; i < 4; i++) v.push_back(Fr::from_uint(l[i]));
    }
    return poseidon_commit_native(v, 0x4D5043ULL);   // "MPC"
}

// Uniform-ish secret scalar from the OS RNG mixed with participant entropy.
inline Fr random_scalar(const std::string& entropy) {
    std::random_device rd;
    std::vector<Fr> v;
    for (int i = 0; i < 8; i++) v.push_back(Fr::from_uint(((uint64_t)rd() << 32) ^ (uint64_t)rd()));
    for (char c : entropy) v.push_back(Fr::from_uint((uint64_t)(unsigned char)c));
    Fr s = poseidon_commit_native(v, 0x53454352ULL);  // "SECR"
    if (s.is_zero()) s = Fr::one();
    return s;
}

// ---------------- contribute / verify ----------------
inline Transcript contribute(PK& pk, VK& vk, const Fr& s, const Fr& k) {
    const G1Affine d_before = pk.delta_g1;
    const Fr s_inv = s.inv();
    pk.delta_g1 = mul1(pk.delta_g1, s);
    pk.delta_g2 = mul2(pk.delta_g2, s);
    vk.delta_g2 = pk.delta_g2;
    for (auto& p : pk.L) p = mul1(p, s_inv);
    for (auto& p : pk.H) p = mul1(p, s_inv);
    Transcript t;
    t.S1 = mul1(G1Affine::generator(), s);
    t.S2 = mul2(G2Affine::generator(), s);
    t.K = mul1(G1Affine::generator(), k);
    Fr c = challenge(t.S1, t.K, d_before, pk.delta_g1);
    t.z = k + c * s;
    return t;
}

inline G1Affine random_combination(const std::vector<G1Affine>& pts, const std::vector<Fr>& rho) {
    G1Jacobian acc = G1Jacobian::from_affine(G1Affine());
    for (size_t i = 0; i < pts.size(); i++) acc = acc + G1Jacobian::from_affine(pts[i]).scalar_mul(rho[i]);
    return acc.to_affine();
}

struct VerifyReport {
    bool pok = false, s_consistent = false, delta_g1 = false, delta_g2 = false, L = false, H = false,
         unchanged = false, vk_ok = false;
    bool ok() const { return pok && s_consistent && delta_g1 && delta_g2 && L && H && unchanged && vk_ok; }
};

inline VerifyReport verify(const PK& before, const PK& after, const VK& vk_after, const Transcript& t) {
    VerifyReport r;
    const G1Affine g1 = G1Affine::generator();
    const G2Affine g2 = G2Affine::generator();
    Fr c = challenge(t.S1, t.K, before.delta_g1, after.delta_g1);
    G1Jacobian rhs = G1Jacobian::from_affine(t.K) + G1Jacobian::from_affine(t.S1).scalar_mul(c);
    r.pok = eq1(mul1(g1, t.z), rhs.to_affine()) && !t.S1.infinity;
    r.s_consistent = bn254::pairing(t.S1, g2) == bn254::pairing(g1, t.S2);
    r.delta_g1 = bn254::pairing(after.delta_g1, g2) == bn254::pairing(before.delta_g1, t.S2);
    r.delta_g2 = bn254::pairing(before.delta_g1, after.delta_g2) == bn254::pairing(after.delta_g1, before.delta_g2);
    // random linear combinations (rho from an independent RNG of the verifier)
    std::mt19937_64 rng(std::random_device{}());
    auto rhos = [&](size_t n) { std::vector<Fr> v(n); for (auto& x : v) x = Fr::from_uint(rng() | 1u); return v; };
    if (before.L.size() == after.L.size()) {
        auto rho = rhos(after.L.size());
        r.L = bn254::pairing(random_combination(after.L, rho), after.delta_g2) ==
              bn254::pairing(random_combination(before.L, rho), before.delta_g2);
    }
    if (before.H.size() == after.H.size()) {
        auto rho = rhos(after.H.size());
        r.H = bn254::pairing(random_combination(after.H, rho), after.delta_g2) ==
              bn254::pairing(random_combination(before.H, rho), before.delta_g2);
    }
    auto same1 = [](const std::vector<G1Affine>& a, const std::vector<G1Affine>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) if (!eq1(a[i], b[i])) return false;
        return true;
    };
    auto same2 = [](const std::vector<G2Affine>& a, const std::vector<G2Affine>& b) {
        if (a.size() != b.size()) return false;
        for (size_t i = 0; i < a.size(); i++) if (!eq2(a[i], b[i])) return false;
        return true;
    };
    r.unchanged = before.num_constraints == after.num_constraints && before.num_variables == after.num_variables &&
                  before.num_public == after.num_public && eq1(before.alpha_g1, after.alpha_g1) &&
                  eq1(before.beta_g1, after.beta_g1) && eq2(before.beta_g2, after.beta_g2) &&
                  eq2(before.gamma_g2, after.gamma_g2) && same1(before.tau_g1, after.tau_g1) &&
                  same2(before.tau_g2, after.tau_g2) && same1(before.A, after.A) && same1(before.B1, after.B1) &&
                  same2(before.B2, after.B2);
    r.vk_ok = eq2(vk_after.delta_g2, after.delta_g2) && eq1(vk_after.alpha_g1, after.alpha_g1) &&
              eq2(vk_after.beta_g2, after.beta_g2) && eq2(vk_after.gamma_g2, after.gamma_g2);
    return r;
}

}  // namespace ceremony
}  // namespace zkml
