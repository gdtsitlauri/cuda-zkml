#include "common.cuh"
#include "prover/groth16.cuh"
#include "verifier/verifier.cuh"
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static bool ends_with_json(const std::string& path) {
    std::string p = to_lower_copy(path);
    return p.size() >= 5 && p.substr(p.size() - 5) == ".json";
}

static std::vector<bn254::Fr> load_public_inputs_json(const std::string& path) {
    std::ifstream in(path);
    if (!in) return {};

    std::string s((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>());

    std::vector<bn254::Fr> out;
    std::string tok;
    auto flush_token = [&]() {
        if (tok.empty() || tok == "-" || tok == "+") {
            tok.clear();
            return;
        }

        bool neg = (tok[0] == '-');
        const char* digits = tok.c_str() + ((tok[0] == '-' || tok[0] == '+') ? 1 : 0);
        unsigned long long v = std::strtoull(digits, nullptr, 10);
        bn254::Fr fr = bn254::Fr::from_uint((uint64_t)v);
        out.push_back(neg ? -fr : fr);
        tok.clear();
    };

    for (char c : s) {
        if ((c >= '0' && c <= '9') || c == '-' || c == '+') {
            if ((c == '-' || c == '+') && !tok.empty()) {
                flush_token();
            }
            tok.push_back(c);
        } else {
            flush_token();
        }
    }
    flush_token();

    return out;
}

// ============================================================
// zkml-verify CLI
//
// Usage:
//   zkml-verify --vk vk.bin --proof proof.bin --public-inputs inputs.bin
//   zkml-verify --vk vk.bin --proof proof.bin --public-inputs inputs.json
//
// Verifies a Groth16 proof using the verification key and public inputs.
// Exits with code 0 if valid, 1 if invalid.
// ============================================================

void print_usage() {
    printf("CUDA-zkML Verifier\n");
    printf("==================\n");
    printf("Usage:\n");
    printf("  zkml-verify --vk VK_FILE --proof PROOF_FILE --public-inputs PI_FILE\n");
    printf("\n");
    printf("Options:\n");
    printf("  --vk FILE           Path to verification key\n");
    printf("  --proof FILE        Path to proof\n");
    printf("  --public-inputs FILE Path to public inputs (.bin or .json)\n");
    printf("  --help              Show this help\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const char* vk_path = nullptr;
    const char* proof_path = nullptr;
    const char* pi_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else if (strcmp(argv[i], "--vk") == 0 && i + 1 < argc) {
            vk_path = argv[++i];
        } else if (strcmp(argv[i], "--proof") == 0 && i + 1 < argc) {
            proof_path = argv[++i];
        } else if (strcmp(argv[i], "--public-inputs") == 0 && i + 1 < argc) {
            pi_path = argv[++i];
        }
    }

    if (!vk_path || !proof_path || !pi_path) {
        fprintf(stderr, "Error: --vk, --proof, and --public-inputs are required\n");
        print_usage();
        return 1;
    }

    // Load verification key
    printf("Loading verification key from %s...\n", vk_path);
    zkml::VerificationKey vk = zkml::load_vk(vk_path);

    // Load proof
    printf("Loading proof from %s...\n", proof_path);
    zkml::Groth16Proof proof = zkml::load_proof(proof_path);

    // Load public inputs (.bin or .json)
    printf("Loading public inputs from %s...\n", pi_path);
    std::vector<bn254::Fr> public_inputs;
    if (ends_with_json(pi_path)) {
        public_inputs = load_public_inputs_json(pi_path);
        if (public_inputs.empty()) {
            fprintf(stderr, "Error: Failed to parse public inputs json: %s\n", pi_path);
            return 1;
        }
    } else {
        public_inputs = zkml::load_public_inputs(pi_path);
    }

    if (!proof.valid) {
        fprintf(stderr, "Error: Invalid proof file\n");
        return 1;
    }

    // Verify
    printf("\nVerifying proof...\n");
    CudaTimer timer;
    timer.start();
    bool valid = zkml::Groth16Verifier::verify(vk, proof, public_inputs);
    float ms = timer.stop();

    printf("\n");
    if (valid) {
        printf("=============================\n");
        printf("   PROOF VALID ✓\n");
        printf("=============================\n");
    } else {
        printf("=============================\n");
        printf("   PROOF INVALID ✗\n");
        printf("=============================\n");
    }
    printf("Verification time: %.2f ms\n", ms);

    return valid ? 0 : 1;
}
