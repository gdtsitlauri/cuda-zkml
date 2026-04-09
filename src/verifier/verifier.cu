#include "verifier/verifier.cuh"
#include "common.cuh"
#include <cstdio>
#include <cstring>

namespace zkml {

// ============================================================
// File I/O for proofs and verification keys
// Binary format: raw bytes, no headers (simple but functional)
// ============================================================

bool save_proof(const Groth16Proof& proof, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "[IO] Cannot open %s for writing\n", filename);
        return false;
    }

    auto write_g1 = [&](const bn254::G1Affine& p) {
        uint64_t std_x[4], std_y[4];
        p.x.to_standard(std_x);
        p.y.to_standard(std_y);
        fwrite(std_x, sizeof(uint64_t), 4, f);
        fwrite(std_y, sizeof(uint64_t), 4, f);
    };

    auto write_g2 = [&](const bn254::G2Affine& p) {
        uint64_t vals[8];
        p.x.c0.to_standard(vals + 0);
        p.x.c1.to_standard(vals + 4);
        fwrite(vals, sizeof(uint64_t), 8, f);
        p.y.c0.to_standard(vals + 0);
        p.y.c1.to_standard(vals + 4);
        fwrite(vals, sizeof(uint64_t), 8, f);
    };

    write_g1(proof.A);
    write_g2(proof.B);
    write_g1(proof.C);
    fclose(f);

    printf("[IO] Saved proof to %s (256 bytes)\n", filename);
    return true;
}

Groth16Proof load_proof(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "[IO] Cannot open %s for reading\n", filename);
        return Groth16Proof();
    }

    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    if (size != 256) {
        fclose(f);
        fprintf(stderr, "[IO] Invalid proof size: %ld bytes\n", size);
        return Groth16Proof();
    }

    auto read_fp = [&]() -> bn254::Fp {
        uint64_t limbs[4];
        size_t got = fread(limbs, sizeof(uint64_t), 4, f);
        if (got != 4) {
            return bn254::Fp::zero();
        }
        return bn254::Fp::from_standard(limbs[0], limbs[1], limbs[2], limbs[3]);
    };

    auto read_g1 = [&]() -> bn254::G1Affine {
        bn254::Fp x = read_fp();
        bn254::Fp y = read_fp();
        return bn254::G1Affine(x, y);
    };

    auto read_g2 = [&]() -> bn254::G2Affine {
        bn254::Fp x0 = read_fp();
        bn254::Fp x1 = read_fp();
        bn254::Fp y0 = read_fp();
        bn254::Fp y1 = read_fp();
        return bn254::G2Affine(bn254::Fp2(x0, x1), bn254::Fp2(y0, y1));
    };

    Groth16Proof proof;
    proof.A = read_g1();
    proof.B = read_g2();
    proof.C = read_g1();
    proof.valid = !feof(f) && !ferror(f);
    fclose(f);

    if (!proof.valid) {
        fprintf(stderr, "[IO] Read error\n");
        return Groth16Proof();
    }

    printf("[IO] Loaded proof from %s (%ld bytes)\n", filename, size);
    return proof;
}

bool save_vk(const VerificationKey& vk, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "[IO] Cannot open %s for writing\n", filename);
        return false;
    }

    // Write number of public inputs
    int32_t num_pub = vk.num_public;
    fwrite(&num_pub, sizeof(int32_t), 1, f);

    // Write alpha_g1 (2 Fp = 64 bytes)
    auto write_g1 = [&](const bn254::G1Affine& p) {
        uint64_t std_x[4], std_y[4];
        p.x.to_standard(std_x);
        p.y.to_standard(std_y);
        fwrite(std_x, sizeof(uint64_t), 4, f);
        fwrite(std_y, sizeof(uint64_t), 4, f);
    };

    auto write_g2 = [&](const bn254::G2Affine& p) {
        uint64_t vals[8];
        p.x.c0.to_standard(vals + 0);
        p.x.c1.to_standard(vals + 4);
        fwrite(vals, sizeof(uint64_t), 8, f);
        p.y.c0.to_standard(vals + 0);
        p.y.c1.to_standard(vals + 4);
        fwrite(vals, sizeof(uint64_t), 8, f);
    };

    write_g1(vk.alpha_g1);
    write_g2(vk.beta_g2);
    write_g2(vk.gamma_g2);
    write_g2(vk.delta_g2);

    // Write IC points
    int32_t num_ic = (int32_t)vk.ic.size();
    fwrite(&num_ic, sizeof(int32_t), 1, f);
    for (const auto& p : vk.ic) {
        write_g1(p);
    }

    fclose(f);
    printf("[IO] Saved verification key to %s\n", filename);
    return true;
}

VerificationKey load_vk(const char* filename) {
    VerificationKey vk;
    FILE* f = fopen(filename, "rb");
    if (!f) {
        fprintf(stderr, "[IO] Cannot open %s\n", filename);
        return vk;
    }

    int32_t num_pub;
    fread(&num_pub, sizeof(int32_t), 1, f);
    vk.num_public = num_pub;

    auto read_fp = [&]() -> bn254::Fp {
        uint64_t limbs[4];
        fread(limbs, sizeof(uint64_t), 4, f);
        return bn254::Fp::from_standard(limbs[0], limbs[1], limbs[2], limbs[3]);
    };

    auto read_g1 = [&]() -> bn254::G1Affine {
        bn254::Fp x = read_fp();
        bn254::Fp y = read_fp();
        return bn254::G1Affine(x, y);
    };

    auto read_g2 = [&]() -> bn254::G2Affine {
        bn254::Fp x0 = read_fp(); bn254::Fp x1 = read_fp();
        bn254::Fp y0 = read_fp(); bn254::Fp y1 = read_fp();
        return bn254::G2Affine(bn254::Fp2(x0, x1), bn254::Fp2(y0, y1));
    };

    vk.alpha_g1 = read_g1();
    vk.beta_g2 = read_g2();
    vk.gamma_g2 = read_g2();
    vk.delta_g2 = read_g2();

    int32_t num_ic;
    fread(&num_ic, sizeof(int32_t), 1, f);
    vk.ic.resize(num_ic);
    for (int i = 0; i < num_ic; i++) {
        vk.ic[i] = read_g1();
    }

    fclose(f);
    printf("[IO] Loaded verification key from %s\n", filename);
    return vk;
}

bool save_public_inputs(const std::vector<bn254::Fr>& inputs, const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) return false;

    int32_t n = (int32_t)inputs.size();
    fwrite(&n, sizeof(int32_t), 1, f);
    for (const auto& inp : inputs) {
        uint64_t std[4];
        inp.to_standard(std);
        fwrite(std, sizeof(uint64_t), 4, f);
    }
    fclose(f);
    return true;
}

std::vector<bn254::Fr> load_public_inputs(const char* filename) {
    std::vector<bn254::Fr> result;
    FILE* f = fopen(filename, "rb");
    if (!f) return result;

    int32_t n;
    fread(&n, sizeof(int32_t), 1, f);
    result.resize(n);
    for (int i = 0; i < n; i++) {
        uint64_t limbs[4];
        fread(limbs, sizeof(uint64_t), 4, f);
        result[i] = bn254::Fr(limbs[0], limbs[1], limbs[2], limbs[3]);
        // Convert to Montgomery form
        bn254::Fr r2 = bn254::Fr::r_squared();
        bn254::Fr::mont_mul_fr(result[i].val, limbs, r2.val);
    }
    fclose(f);
    return result;
}

} // namespace zkml
