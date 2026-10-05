// zkml-ceremony: Groth16 phase-2 MPC contributions (roadmap 2.5).
//
//   zkml-ceremony contribute --pk-in pk.bin --vk-in vk.bin --pk-out pk1.bin --vk-out vk1.bin
//                            --transcript t1.bin [--entropy "free text"]
//   zkml-ceremony verify     --pk-before pk.bin --pk-after pk1.bin --vk-after vk1.bin --transcript t1.bin
//
// Chain: setup (zkml-prove --pk-save) -> participant 1 -> ... -> participant N.
// Each participant runs `contribute` and must then destroy its process memory
// (the secret never touches disk). Anyone can `verify` every step; the final
// pk/vk are used with zkml-prove --pk-load / zkml-verify. Host-only code.

#include "prover/ceremony.cuh"
#include <cstdio>
#include <cstring>
#include <string>

using namespace zkml;

static const char* arg(int argc, char** argv, const char* name) {
    for (int i = 2; i + 1 < argc; i++) if (std::strcmp(argv[i], name) == 0) return argv[i + 1];
    return nullptr;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: zkml-ceremony contribute|verify ... (see source header)\n");
        return 1;
    }
    std::string mode = argv[1];
    if (mode == "contribute") {
        const char *pki = arg(argc, argv, "--pk-in"), *vki = arg(argc, argv, "--vk-in");
        const char *pko = arg(argc, argv, "--pk-out"), *vko = arg(argc, argv, "--vk-out");
        const char *tr = arg(argc, argv, "--transcript"), *ent = arg(argc, argv, "--entropy");
        if (!pki || !vki || !pko || !vko || !tr) { fprintf(stderr, "missing arguments\n"); return 1; }
        ceremony::PK pk;
        ceremony::VK vk;
        if (!ceremony::load_pk(pki, pk) || !ceremony::load_vk(vki, vk)) { fprintf(stderr, "cannot load keys\n"); return 1; }
        Fr s = ceremony::random_scalar(ent ? ent : "");
        Fr k = ceremony::random_scalar(std::string("nonce:") + (ent ? ent : ""));
        ceremony::Transcript t = ceremony::contribute(pk, vk, s, k);
        s = Fr::zero();
        k = Fr::zero();
        if (!ceremony::save_pk(pko, pk) || !ceremony::save_vk(vko, vk) || !ceremony::save_transcript(tr, t)) {
            fprintf(stderr, "cannot write outputs\n");
            return 1;
        }
        printf("[MPC] contribution written: %s %s %s (secret discarded)\n", pko, vko, tr);
        return 0;
    }
    if (mode == "verify") {
        const char *pb = arg(argc, argv, "--pk-before"), *pa = arg(argc, argv, "--pk-after");
        const char *va = arg(argc, argv, "--vk-after"), *tr = arg(argc, argv, "--transcript");
        if (!pb || !pa || !va || !tr) { fprintf(stderr, "missing arguments\n"); return 1; }
        ceremony::PK before, after;
        ceremony::VK vk;
        ceremony::Transcript t;
        if (!ceremony::load_pk(pb, before) || !ceremony::load_pk(pa, after) || !ceremony::load_vk(va, vk) ||
            !ceremony::load_transcript(tr, t)) {
            fprintf(stderr, "cannot load inputs\n");
            return 1;
        }
        ceremony::VerifyReport r = ceremony::verify(before, after, vk, t);
        printf("[MPC] pok=%d s=%d delta_g1=%d delta_g2=%d L=%d H=%d unchanged=%d vk=%d\n", r.pok, r.s_consistent,
               r.delta_g1, r.delta_g2, r.L, r.H, r.unchanged, r.vk_ok);
        printf(r.ok() ? "[MPC] CONTRIBUTION VALID\n" : "[MPC] CONTRIBUTION INVALID\n");
        return r.ok() ? 0 : 1;
    }
    fprintf(stderr, "unknown mode %s\n", mode.c_str());
    return 1;
}
