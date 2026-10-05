#pragma once

// Export of an R1CS (and optionally a witness) in the iden3 binary formats used by
// circom / snarkjs (.r1cs v1, .wtns v2), roadmap 1.2. This lets external
// underconstraint tools (e.g. Picus, snarkjs r1cs info) and
// tests/test_underconstrained.py analyse exactly the circuit the prover uses.
//
// Wire mapping: wire 0 = constant 1, wires 1..num_public = public inputs
// (exported as public outputs, nPubIn = 0), the rest private; label i = wire i.

#include "prover/circuit.cuh"
#include <cstdio>
#include <vector>

namespace zkml {

namespace r1cs_io {

inline void put_u32(std::vector<uint8_t>& b, uint32_t v) { for (int i = 0; i < 4; i++) b.push_back((uint8_t)(v >> (8 * i))); }
inline void put_u64(std::vector<uint8_t>& b, uint64_t v) { for (int i = 0; i < 8; i++) b.push_back((uint8_t)(v >> (8 * i))); }
inline void put_fr(std::vector<uint8_t>& b, const Fr& x) {
    uint64_t s[4];
    x.to_standard(s);
    for (int k = 0; k < 4; k++) put_u64(b, s[k]);
}
inline void put_prime(std::vector<uint8_t>& b) {
    uint64_t r[4];
    Fr::get_r(r);
    for (int k = 0; k < 4; k++) put_u64(b, r[k]);
}
inline bool write_file(const char* path, const std::vector<uint8_t>& b) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite(b.data(), 1, b.size(), f) == b.size();
    fclose(f);
    return ok;
}

}  // namespace r1cs_io

inline bool export_r1cs_iden3(const R1CS& c, const char* path) {
    using namespace r1cs_io;
    const int m = c.num_constraints;
    std::vector<std::vector<std::pair<int, Fr>>> A((size_t)m), Bm((size_t)m), C((size_t)m);
    for (auto& e : c.A) A[(size_t)e.row].push_back({e.col, e.value});
    for (auto& e : c.B) Bm[(size_t)e.row].push_back({e.col, e.value});
    for (auto& e : c.C) C[(size_t)e.row].push_back({e.col, e.value});

    std::vector<uint8_t> hdr, cons, map, out;
    put_u32(hdr, 32);
    put_prime(hdr);
    put_u32(hdr, (uint32_t)c.num_variables);         // nWires
    put_u32(hdr, (uint32_t)c.num_public_inputs);     // nPubOut
    put_u32(hdr, 0);                                 // nPubIn
    put_u32(hdr, (uint32_t)c.num_private_inputs);    // nPrvIn
    put_u64(hdr, (uint64_t)c.num_variables);         // nLabels
    put_u32(hdr, (uint32_t)m);                       // mConstraints
    auto put_lc = [&](const std::vector<std::pair<int, Fr>>& lc) {
        put_u32(cons, (uint32_t)lc.size());
        for (auto& [w, v] : lc) { put_u32(cons, (uint32_t)w); put_fr(cons, v); }
    };
    for (int i = 0; i < m; i++) { put_lc(A[(size_t)i]); put_lc(Bm[(size_t)i]); put_lc(C[(size_t)i]); }
    for (int i = 0; i < c.num_variables; i++) put_u64(map, (uint64_t)i);

    out.insert(out.end(), {'r', '1', 'c', 's'});
    put_u32(out, 1);   // version
    put_u32(out, 3);   // sections
    auto section = [&](uint32_t type, const std::vector<uint8_t>& body) {
        put_u32(out, type);
        put_u64(out, (uint64_t)body.size());
        out.insert(out.end(), body.begin(), body.end());
    };
    section(1, hdr);
    section(2, cons);
    section(3, map);
    return write_file(path, out);
}

inline bool export_wtns_iden3(const std::vector<Fr>& w, const char* path) {
    using namespace r1cs_io;
    std::vector<uint8_t> hdr, vals, out;
    put_u32(hdr, 32);
    put_prime(hdr);
    put_u32(hdr, (uint32_t)w.size());
    for (auto& x : w) put_fr(vals, x);
    out.insert(out.end(), {'w', 't', 'n', 's'});
    put_u32(out, 2);
    put_u32(out, 2);
    put_u32(out, 1); put_u64(out, (uint64_t)hdr.size()); out.insert(out.end(), hdr.begin(), hdr.end());
    put_u32(out, 2); put_u64(out, (uint64_t)vals.size()); out.insert(out.end(), vals.begin(), vals.end());
    return write_file(path, out);
}

}  // namespace zkml
