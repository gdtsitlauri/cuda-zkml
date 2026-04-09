#include "common.cuh"
#include "prover/groth16.cuh"
#include "prover/witness.cuh"
#include "verifier/verifier.cuh"
#include "nn/inference.cuh"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cctype>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// ============================================================
// zkml-prove CLI
//
// Usage:
//   zkml-prove --model model.bin --input input.bin --output proof.bin
//              [--vk vk.bin] [--public-inputs inputs.bin]
//
// Or for demo mode:
//   zkml-prove --demo
//
// This tool:
// 1. Loads a quantized neural network model
// 2. Loads input data
// 3. Runs inference in finite field on GPU
// 4. Generates R1CS circuit from NN structure
// 5. Generates witness from inference trace
// 6. Runs Groth16 trusted setup (or loads existing keys)
// 7. Generates Groth16 proof using GPU-accelerated MSM
// 8. Saves proof and verification key to files
// ============================================================

static std::string to_lower_copy(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

static std::string file_ext_lower(const std::string& path) {
    size_t pos = path.find_last_of('.');
    if (pos == std::string::npos) return "";
    return to_lower_copy(path.substr(pos));
}

static std::string quote_arg(const std::string& arg) {
    std::string escaped;
    escaped.reserve(arg.size() + 2);
    for (char c : arg) {
        if (c == '"') escaped.push_back('\\');
        escaped.push_back(c);
    }
    return std::string("\"") + escaped + "\"";
}

static bool file_exists(const char* path) {
    if (!path) return false;
    std::ifstream in(path, std::ios::binary);
    return (bool)in;
}

static bool file_exists_str(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return (bool)in;
}

static bool parse_layer_type(const std::string& type, zkml::LayerType& out) {
    std::string lower = to_lower_copy(type);
    if (lower == "linear") {
        out = zkml::LayerType::LINEAR;
        return true;
    }
    if (lower == "relu" || lower == "relu_approx") {
        out = zkml::LayerType::RELU_APPROX;
        return true;
    }
    if (lower == "softmax" || lower == "softmax_approx") {
        out = zkml::LayerType::SOFTMAX_APPROX;
        return true;
    }
    if (lower == "self_attention" || lower == "attention" || lower == "transformer_attention") {
        out = zkml::LayerType::SELF_ATTENTION;
        return true;
    }
    return false;
}

static bool extract_json_string_field(const std::string& object_text,
                                      const char* field,
                                      std::string& out) {
    std::regex re(std::string("\"") + field + "\"\\s*:\\s*\"([^\"]+)\"");
    std::smatch match;
    if (!std::regex_search(object_text, match, re) || match.size() < 2) {
        return false;
    }
    out = match[1].str();
    return true;
}

static bool extract_json_int_field(const std::string& object_text,
                                   const char* field,
                                   int& out) {
    std::regex re(std::string("\"") + field + "\"\\s*:\\s*([0-9]+)");
    std::smatch match;
    if (!std::regex_search(object_text, match, re) || match.size() < 2) {
        return false;
    }
    out = std::atoi(match[1].str().c_str());
    return true;
}

static bool split_layer_objects(const std::string& text,
                                std::vector<std::string>& objects) {
    objects.clear();
    size_t layers_pos = text.find("\"layers\"");
    if (layers_pos == std::string::npos) {
        return false;
    }
    size_t array_start = text.find('[', layers_pos);
    if (array_start == std::string::npos) {
        return false;
    }

    int depth = 0;
    size_t object_start = std::string::npos;
    for (size_t i = array_start; i < text.size(); i++) {
        char c = text[i];
        if (c == '{') {
            if (depth == 0) {
                object_start = i;
            }
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0 && object_start != std::string::npos) {
                objects.push_back(text.substr(object_start, i - object_start + 1));
                object_start = std::string::npos;
            }
        } else if (c == ']' && depth == 0) {
            break;
        }
    }
    return !objects.empty();
}

static bool load_architecture_json(const std::string& path,
                                   std::vector<zkml::NNModel::LayerSpec>& specs) {
    std::ifstream in(path);
    if (!in) return false;

    std::stringstream buffer;
    buffer << in.rdbuf();
    const std::string text = buffer.str();

    specs.clear();
    std::vector<std::string> layer_objects;
    if (!split_layer_objects(text, layer_objects)) {
        fprintf(stderr, "[Model] Could not parse layers array from architecture sidecar: %s\n",
                path.c_str());
        return false;
    }

    for (const auto& object_text : layer_objects) {
        std::string type_text;
        int in_size = 0;
        int out_size = 0;
        int seq_len = 0;
        int hidden_size = 0;
        int num_heads = 1;
        if (!extract_json_string_field(object_text, "type", type_text) ||
            !extract_json_int_field(object_text, "in_features", in_size) ||
            !extract_json_int_field(object_text, "out_features", out_size)) {
            fprintf(stderr, "[Model] Invalid layer object in architecture sidecar\n");
            return false;
        }

        zkml::LayerType type;
        if (!parse_layer_type(type_text, type)) {
            fprintf(stderr, "[Model] Unsupported layer type in architecture: %s\n",
                    type_text.c_str());
            return false;
        }

        if (in_size <= 0 || out_size <= 0) {
            fprintf(stderr, "[Model] Invalid layer dimensions in architecture sidecar\n");
            return false;
        }

        if (type == zkml::LayerType::SELF_ATTENTION) {
            if (!extract_json_int_field(object_text, "seq_len", seq_len) ||
                !extract_json_int_field(object_text, "hidden_size", hidden_size) ||
                seq_len <= 0 || hidden_size <= 0) {
                fprintf(stderr,
                        "[Model] Self-attention layers require seq_len and hidden_size metadata\n");
                return false;
            }
            if (extract_json_int_field(object_text, "num_heads", num_heads) && num_heads <= 0) {
                fprintf(stderr, "[Model] Self-attention num_heads must be positive\n");
                return false;
            }
        }

        specs.push_back({type, in_size, out_size, seq_len, hidden_size, num_heads});
    }

    if (specs.empty()) {
        fprintf(stderr, "[Model] No layers found in architecture sidecar: %s\n", path.c_str());
        return false;
    }

    for (size_t i = 0; i < specs.size(); i++) {
        if (specs[i].type == zkml::LayerType::SOFTMAX_APPROX && i + 1 != specs.size()) {
            fprintf(stderr,
                    "[Model] Only final softmax layers are supported in architecture sidecars\n");
            return false;
        }
    }

    return true;
}

static bool load_model_from_architecture(const std::string& arch_path,
                                         zkml::NNModel& model) {
    std::vector<zkml::NNModel::LayerSpec> specs;
    if (!load_architecture_json(arch_path, specs)) {
        return false;
    }
    model = zkml::NNModel::create_from_specs(specs);
    return !model.layers.empty();
}

static bool create_default_model_for_input_size(int input_size,
                                                zkml::NNModel& model) {
    if (input_size == 32) {
        model = zkml::NNModel::create_tiny_transformer();
        return true;
    }
    if (input_size == 784) {
        model = zkml::NNModel::create_mnist_mlp();
        return true;
    }
    if (input_size == 3072) {
        model = zkml::NNModel::create_cifar_mlp();
        return true;
    }
    fprintf(stderr,
            "[Model] No default architecture for input size %d. Pass --arch MODEL.arch.json\n",
            input_size);
    return false;
}

static bool load_float32_binary(const std::string& path, std::vector<float>& out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (sz <= 0 || (sz % (long)sizeof(float)) != 0) {
        fclose(f);
        return false;
    }
    out.resize((size_t)sz / sizeof(float));
    size_t n = fread(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    return n == out.size();
}

static bool load_npy_float32(const std::string& path, std::vector<float>& out) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;

    char magic[6] = {0};
    in.read(magic, 6);
    if (!in || std::memcmp(magic, "\x93NUMPY", 6) != 0) return false;

    uint8_t major = 0, minor = 0;
    in.read(reinterpret_cast<char*>(&major), 1);
    in.read(reinterpret_cast<char*>(&minor), 1);
    if (!in) return false;

    uint32_t header_len = 0;
    if (major == 1) {
        uint16_t h16 = 0;
        in.read(reinterpret_cast<char*>(&h16), 2);
        header_len = h16;
    } else if (major == 2 || major == 3) {
        in.read(reinterpret_cast<char*>(&header_len), 4);
    } else {
        return false;
    }
    if (!in || header_len == 0) return false;

    std::string header(header_len, '\0');
    in.read(&header[0], header_len);
    if (!in) return false;

    if (header.find("<f4") == std::string::npos && header.find("|f4") == std::string::npos) {
        return false;
    }
    if (header.find("fortran_order") != std::string::npos &&
        header.find("True") != std::string::npos) {
        return false; // only C-contiguous arrays are supported
    }

    size_t p0 = header.find('(');
    size_t p1 = header.find(')', p0 == std::string::npos ? 0 : p0 + 1);
    if (p0 == std::string::npos || p1 == std::string::npos || p1 <= p0 + 1) return false;

    std::string shape = header.substr(p0 + 1, p1 - p0 - 1);
    std::stringstream ss(shape);
    std::string token;
    std::vector<size_t> dims;
    while (std::getline(ss, token, ',')) {
        std::string t;
        for (char c : token) {
            if (!std::isspace((unsigned char)c)) t.push_back(c);
        }
        if (t.empty()) continue;
        long long d = std::atoll(t.c_str());
        if (d <= 0) return false;
        dims.push_back((size_t)d);
    }
    if (dims.empty()) return false;

    size_t n = 1;
    for (size_t d : dims) {
        if (d > (SIZE_MAX / n)) return false;
        n *= d;
    }

    out.resize(n);
    in.read(reinterpret_cast<char*>(out.data()), (std::streamsize)(n * sizeof(float)));
    return (bool)in;
}

static bool load_input_float32_auto(const std::string& path, std::vector<float>& out) {
    std::string ext = file_ext_lower(path);
    if (ext == ".npy") {
        return load_npy_float32(path, out);
    }
    return load_float32_binary(path, out);
}

static bool save_public_inputs_json(const std::vector<bn254::Fr>& inputs,
                                    const std::string& path) {
    std::ofstream out(path);
    if (!out) return false;

    out << "[";
    for (size_t i = 0; i < inputs.size(); i++) {
        uint64_t limbs[4] = {0, 0, 0, 0};
        inputs[i].to_standard(limbs);
        out << limbs[0]; // low-limb decimal for lightweight interoperability
        if (i + 1 < inputs.size()) out << ", ";
    }
    out << "]\n";
    return true;
}

static bn254::Fr fp_to_fr_scalar(const bn254::Fp& v) {
    uint64_t std_val[4];
    v.to_standard(std_val);
    bn254::Fr fr_val;
    bn254::Fr::mont_mul_fr(fr_val.val, std_val, bn254::Fr::r_squared().val);
    return fr_val;
}

static bool build_circuit_for_model(zkml::CircuitBuilder& builder,
                                    const zkml::NNModel& model,
                                    int input_size) {
    if (model.layers.empty()) {
        fprintf(stderr, "[Circuit] Model has no layers\n");
        return false;
    }

    int public_output_size = zkml::circuit_public_output_size(model);
    if (public_output_size <= 0) {
        fprintf(stderr, "[Circuit] Could not determine circuit public outputs\n");
        return false;
    }

    std::vector<int> public_output_vars(public_output_size);
    for (int i = 0; i < public_output_size; i++) {
        public_output_vars[i] = builder.alloc_var();
    }

    std::vector<int> current_vars(input_size);
    for (int i = 0; i < input_size; i++) {
        current_vars[i] = builder.alloc_var();
    }

    for (int layer_idx = 0; layer_idx < (int)model.layers.size(); layer_idx++) {
        const auto& layer = model.layers[layer_idx];

        if (layer.type == zkml::LayerType::LINEAR) {
            if ((int)current_vars.size() != layer.in_size) {
                fprintf(stderr, "[Circuit] Linear layer size mismatch: got %d, expected %d\n",
                        (int)current_vars.size(), layer.in_size);
                return false;
            }

            bool use_public_outputs = zkml::is_last_circuit_layer(model, layer_idx) &&
                                      layer.out_size == public_output_size;
            if (use_public_outputs) {
                current_vars = builder.add_linear_layer_into(
                    current_vars, public_output_vars, layer.out_size, layer.in_size);
            } else {
                current_vars = builder.add_linear_layer(current_vars, layer.out_size, layer.in_size);
            }
        } else if (layer.type == zkml::LayerType::RELU_APPROX) {
            if ((int)current_vars.size() != layer.in_size) {
                fprintf(stderr, "[Circuit] ReLU layer size mismatch: got %d, expected %d\n",
                        (int)current_vars.size(), layer.in_size);
                return false;
            }
            current_vars = builder.add_relu_approx(
                current_vars,
                layer.in_size,
                fp_to_fr_scalar(layer.relu_params.c0),
                fp_to_fr_scalar(layer.relu_params.c1),
                fp_to_fr_scalar(layer.relu_params.c2),
                fp_to_fr_scalar(layer.relu_params.c3),
                fp_to_fr_scalar(layer.relu_params.scale));
        } else if (layer.type == zkml::LayerType::SELF_ATTENTION) {
            if ((int)current_vars.size() != layer.in_size) {
                fprintf(stderr, "[Circuit] Attention layer size mismatch: got %d, expected %d\n",
                        (int)current_vars.size(), layer.in_size);
                return false;
            }
            current_vars = builder.add_self_attention_layer(
                current_vars,
                layer.attention_params.seq_len,
                layer.attention_params.hidden_size,
                layer.attention_params.num_heads);
        }
    }

    builder.finalize(public_output_size, input_size);
    return true;
}

static bool resolve_model_to_binary(const std::string& input_model_path,
                                    std::string& resolved_model_bin_path) {
    std::string ext = file_ext_lower(input_model_path);
    if (ext == ".bin") {
        resolved_model_bin_path = input_model_path;
        return true;
    }

    if (ext == ".onnx") {
        resolved_model_bin_path = input_model_path + ".weights.bin";

        std::string script_path = "python/convert_model.py";
        {
            FILE* f = fopen(script_path.c_str(), "rb");
            if (!f) {
                script_path = "../python/convert_model.py";
                f = fopen(script_path.c_str(), "rb");
            }
            if (f) fclose(f);
            if (!f) {
                fprintf(stderr,
                        "[Model] Could not find converter script (python/convert_model.py)\n");
                return false;
            }
        }

        std::string cmd =
            std::string("python ") +
            quote_arg(script_path) +
            " --input " + quote_arg(input_model_path) +
            " --output " + quote_arg(resolved_model_bin_path);

        printf("[Model] Converting ONNX to binary weights...\n");
        int rc = std::system(cmd.c_str());
        if (rc != 0) {
            fprintf(stderr, "[Model] ONNX conversion failed (exit=%d)\n", rc);
            return false;
        }
        return true;
    }

    fprintf(stderr, "Unsupported model format: %s (expected .bin or .onnx)\n",
            input_model_path.c_str());
    return false;
}

static bool resolve_architecture_path(const char* arch_path_arg,
                                      const std::string& original_model_path,
                                      const std::string& resolved_model_bin_path,
                                      std::string& arch_path_out) {
    if (arch_path_arg && file_exists(arch_path_arg)) {
        arch_path_out = arch_path_arg;
        return true;
    }

    std::vector<std::string> candidates = {
        resolved_model_bin_path + ".arch.json",
        original_model_path + ".arch.json",
    };

    for (const auto& candidate : candidates) {
        if (file_exists_str(candidate)) {
            arch_path_out = candidate;
            return true;
        }
    }

    arch_path_out.clear();
    return false;
}

void print_usage() {
    printf("CUDA-zkML Prover\n");
    printf("================\n");
    printf("Usage:\n");
    printf("  zkml-prove --model MODEL --input INPUT --output PROOF\n");
    printf("             [--vk VK_FILE] [--public-inputs PI_FILE]\n");
    printf("             [--pk-save PK_FILE] [--pk-load PK_FILE] [--arch ARCH_JSON]\n");
    printf("  zkml-prove --demo\n");
    printf("\n");
    printf("Options:\n");
    printf("  --model FILE        Path to model (.bin or .onnx)\n");
    printf("  --input FILE        Path to input (.bin or .npy float32)\n");
    printf("  --output FILE       Path to write proof\n");
    printf("  --vk FILE           Path to write verification key\n");
    printf("  --public-inputs FILE Path to write public inputs\n");
    printf("  --arch FILE         Optional model architecture sidecar (.arch.json)\n");
    printf("  --pk-save FILE      Save proving key after setup for reuse\n");
    printf("  --pk-load FILE      Load proving key from disk instead of setup\n");
    printf("  --demo              Run demo with random MNIST MLP\n");
    printf("  --help              Show this help\n");
}

int run_demo() {
    printf("=== CUDA-zkML Demo: Proving MNIST MLP Inference ===\n\n");

    // Print GPU info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d, %d SMs, %.0f MB VRAM)\n",
           prop.name, prop.major, prop.minor,
           prop.multiProcessorCount,
           prop.totalGlobalMem / (1024.0 * 1024.0));
    print_gpu_memory("Initial");

    // Create MNIST MLP model
    printf("\n[1/5] Creating MNIST MLP model (784 → 128 → 10)...\n");
    zkml::NNModel model = zkml::NNModel::create_mnist_mlp();
    model.init_random(42);

    // Create random input (simulating MNIST image)
    printf("[2/5] Generating random input (784 pixels)...\n");
    int input_size = 784;
    std::vector<float> float_input(input_size);
    srand(42);
    for (int i = 0; i < input_size; i++) {
        float_input[i] = (float)(rand() % 256) / 255.0f;
    }

    // Quantize input
    float max_abs = zkml::find_max_abs(float_input.data(), input_size);
    zkml::QuantConfig input_qcfg = zkml::QuantConfig::symmetric(8, max_abs);
    std::vector<bn254::Fp> fp_input(input_size);
    zkml::quantize_array(fp_input.data(), float_input.data(), input_size, input_qcfg);

    // Run inference
    printf("[3/5] Running quantized inference on GPU...\n");
    CudaTimer timer;
    timer.start();
    zkml::InferenceTrace trace = zkml::run_inference_gpu(model, fp_input.data(), input_size);
    float inference_ms = timer.stop();
    printf("  Inference time: %.2f ms\n", inference_ms);

    if (!trace.valid) {
        fprintf(stderr, "Inference failed!\n");
        return 1;
    }

    // Print output
    printf("  Output (10 classes):\n  ");
    for (int i = 0; i < (int)trace.final_output.size(); i++) {
        trace.final_output[i].print("");
        printf("  ");
    }
    printf("\n");

    // Generate witness
    printf("[4/5] Generating witness and R1CS circuit...\n");
    zkml::Witness witness = zkml::generate_witness(model, trace, fp_input.data(), input_size);
    printf("  Witness size: %d values\n", (int)witness.values.size());

    // Build R1CS circuit
    zkml::CircuitBuilder builder;
    if (!build_circuit_for_model(builder, model, input_size)) {
        fprintf(stderr, "Failed to build R1CS circuit for model.\n");
        return 1;
    }
    printf("  R1CS: %d constraints, %d variables\n",
           builder.circuit.num_constraints, builder.circuit.num_variables);

    bool sat = builder.circuit.verify_witness(witness.values);
    printf("  Witness satisfies R1CS: %s\n", sat ? "YES" : "NO");
    if (!sat) {
        fprintf(stderr, "Witness does not satisfy circuit constraints.\n");
        return 1;
    }

    // Groth16 setup and proving
    printf("[5/5] Running Groth16 proof generation (GPU-accelerated)...\n");
    zkml::ProvingKey pk;
    zkml::VerificationKey vk;

    timer.start();
    zkml::Groth16Prover::setup(builder.circuit, pk, vk);
    float setup_ms = timer.stop();
    printf("  Setup time: %.2f ms\n", setup_ms);

    timer.start();
    zkml::Groth16Proof proof = zkml::Groth16Prover::prove(pk, builder.circuit, witness);
    float prove_ms = timer.stop();
    printf("  Proving time: %.2f ms\n", prove_ms);

    // Verify
    std::vector<bn254::Fr> pub_inputs = witness.get_public();
    timer.start();
    bool valid = zkml::Groth16Verifier::verify(vk, proof, pub_inputs);
    float verify_ms = timer.stop();

    printf("\n=== Results ===\n");
    printf("Inference time:     %.2f ms\n", inference_ms);
    printf("Setup time:         %.2f ms\n", setup_ms);
    printf("Proving time:       %.2f ms\n", prove_ms);
    printf("Verification time:  %.2f ms\n", verify_ms);
    printf("Proof size:         %zu bytes\n", proof.serialize().size());
    printf("Proof valid:        %s\n", valid ? "YES" : "NO");

    // Save proof
    zkml::save_proof(proof, "proof.bin");
    zkml::save_vk(vk, "vk.bin");
    zkml::save_public_inputs(pub_inputs, "public_inputs.bin");
    save_public_inputs_json(pub_inputs, "public_inputs.json");

    printf("\nOutput files:\n");
    printf("  proof.bin          - Groth16 proof\n");
    printf("  vk.bin             - Verification key\n");
    printf("  public_inputs.bin  - Public inputs\n");
    printf("\nPROOF GENERATION COMPLETE.\n");

    return 0;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    // Parse arguments
    bool demo = false;
    const char* model_path = nullptr;
    const char* input_path = nullptr;
    const char* output_path = "proof.bin";
    const char* vk_path = "vk.bin";
    const char* pi_path = "public_inputs.bin";
    const char* arch_path = nullptr;
    const char* pk_save_path = nullptr;
    const char* pk_load_path = nullptr;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--demo") == 0) {
            demo = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            print_usage();
            return 0;
        } else if (strcmp(argv[i], "--model") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        } else if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) {
            input_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_path = argv[++i];
        } else if (strcmp(argv[i], "--vk") == 0 && i + 1 < argc) {
            vk_path = argv[++i];
        } else if (strcmp(argv[i], "--public-inputs") == 0 && i + 1 < argc) {
            pi_path = argv[++i];
        } else if (strcmp(argv[i], "--arch") == 0 && i + 1 < argc) {
            arch_path = argv[++i];
        } else if (strcmp(argv[i], "--pk-save") == 0 && i + 1 < argc) {
            pk_save_path = argv[++i];
        } else if (strcmp(argv[i], "--pk-load") == 0 && i + 1 < argc) {
            pk_load_path = argv[++i];
        }
    }

    if (demo) {
        return run_demo();
    }

    if (!model_path || !input_path) {
        fprintf(stderr, "Error: --model and --input are required\n");
        print_usage();
        return 1;
    }

    // Resolve model path to binary if needed.
    std::string resolved_model_path;
    if (!resolve_model_to_binary(model_path, resolved_model_path)) {
        return 1;
    }

    // Load model weights.
    printf("Loading model from %s...\n", resolved_model_path.c_str());
    std::vector<float> model_data;
    if (!load_float32_binary(resolved_model_path, model_data)) {
        fprintf(stderr, "Cannot load model file: %s\n", resolved_model_path.c_str());
        return 1;
    }
    int num_floats = (int)model_data.size();

    // Load input (.bin or .npy float32)
    printf("Loading input from %s...\n", input_path);
    std::vector<float> float_input;
    if (!load_input_float32_auto(input_path, float_input)) {
        fprintf(stderr, "Cannot load input file: %s\n", input_path);
        return 1;
    }
    int input_size = (int)float_input.size();

    // Create model topology and load weights.
    zkml::NNModel model;
    std::string resolved_arch_path;
    bool has_arch = resolve_architecture_path(arch_path, model_path, resolved_model_path, resolved_arch_path);
    if (has_arch) {
        printf("Loading architecture from %s...\n", resolved_arch_path.c_str());
        if (!load_model_from_architecture(resolved_arch_path, model)) {
            fprintf(stderr, "Failed to load model architecture\n");
            return 1;
        }
    } else if (!create_default_model_for_input_size(input_size, model)) {
        return 1;
    }

    if (!model.load_weights(model_data.data(), num_floats)) {
        fprintf(stderr, "Failed to load model weights\n");
        return 1;
    }

    // Quantize input
    float max_abs = zkml::find_max_abs(float_input.data(), input_size);
    zkml::QuantConfig qcfg = zkml::QuantConfig::symmetric(8, max_abs);
    std::vector<bn254::Fp> fp_input(input_size);
    zkml::quantize_array(fp_input.data(), float_input.data(), input_size, qcfg);

    CudaTimer timer;

    // Run inference + witness generation using the same split path as demo mode so
    // custom models emit comparable timing metrics and exercise the traced witness flow.
    timer.start();
    zkml::InferenceTrace trace = zkml::run_inference_gpu(model, fp_input.data(), input_size);
    float inference_ms = timer.stop();
    if (!trace.valid) {
        fprintf(stderr, "Inference failed.\n");
        return 1;
    }

    zkml::Witness witness = zkml::generate_witness(model, trace, fp_input.data(), input_size);

    zkml::CircuitBuilder builder;
    if (!build_circuit_for_model(builder, model, input_size)) {
        return 1;
    }

    bool sat = builder.circuit.verify_witness(witness.values);
    if (!sat) {
        fprintf(stderr, "Witness does not satisfy circuit constraints.\n");
        return 1;
    }

    zkml::ProvingKey pk;
    zkml::VerificationKey vk;
    if (pk_load_path) {
        printf("Loading proving key from %s...\n", pk_load_path);
        if (!pk.load_streaming(pk_load_path)) {
            fprintf(stderr, "Failed to load proving key: %s\n", pk_load_path);
            return 1;
        }

        if (pk.A_query_scalars.empty() && !pk.materialized_points) {
            fprintf(stderr,
                    "Loaded proving key does not contain usable scalar or point queries.\n");
            return 1;
        }

        if (!file_exists(vk_path)) {
            fprintf(stderr,
                    "When using --pk-load, a verification key file must already exist at %s\n",
                    vk_path);
            return 1;
        }

        printf("Loading verification key from %s...\n", vk_path);
        vk = zkml::load_vk(vk_path);
    } else {
        timer.start();
        zkml::Groth16Prover::setup(builder.circuit, pk, vk);
        float setup_ms = timer.stop();
        if (pk_save_path) {
            if (!pk.save(pk_save_path)) {
                fprintf(stderr, "Failed to save proving key: %s\n", pk_save_path);
                return 1;
            }
        }
        printf("Setup time: %.2f ms\n", setup_ms);
    }

    timer.start();
    zkml::Groth16Proof proof = zkml::Groth16Prover::prove(pk, builder.circuit, witness);
    float prove_ms = timer.stop();
    std::vector<bn254::Fr> pub = witness.get_public();
    timer.start();
    bool valid = zkml::Groth16Verifier::verify(vk, proof, pub);
    float verify_ms = timer.stop();

    printf("Proof valid: %s\n", valid ? "YES" : "NO");
    if (!valid) {
        fprintf(stderr, "Generated proof failed native verification.\n");
        return 1;
    }

    zkml::save_proof(proof, output_path);
    zkml::save_vk(vk, vk_path);
    if (file_ext_lower(pi_path) == ".json") {
        if (!save_public_inputs_json(pub, pi_path)) {
            fprintf(stderr, "Failed to save public inputs json: %s\n", pi_path);
            return 1;
        }
    } else {
        zkml::save_public_inputs(pub, pi_path);
    }

    // Validate the on-disk artifacts immediately so file-format regressions are
    // caught by the prover before a separate verifier process is launched.
    if (file_ext_lower(pi_path) != ".json") {
        zkml::Groth16Proof loaded_proof = zkml::load_proof(output_path);
        zkml::VerificationKey loaded_vk = zkml::load_vk(vk_path);
        std::vector<bn254::Fr> loaded_pub = zkml::load_public_inputs(pi_path);
        bool roundtrip_ok = zkml::Groth16Verifier::verify(loaded_vk, loaded_proof, loaded_pub);
        if (!roundtrip_ok) {
            bool proof_only_ok = zkml::Groth16Verifier::verify(vk, loaded_proof, pub);
            bool vk_only_ok = zkml::Groth16Verifier::verify(loaded_vk, proof, pub);
            bool public_only_ok = zkml::Groth16Verifier::verify(vk, proof, loaded_pub);
            fprintf(stderr,
                    "Artifact roundtrip mismatch: proof=%s vk=%s public_inputs=%s\n",
                    proof_only_ok ? "ok" : "bad",
                    vk_only_ok ? "ok" : "bad",
                    public_only_ok ? "ok" : "bad");
            return 1;
        }
    }

    printf("Inference time: %.2f ms\n", inference_ms);
    if (pk_load_path) {
        printf("Setup time: %.2f ms\n", 0.0f);
    }
    printf("Proving time: %.2f ms\n", prove_ms);
    printf("Verification time: %.2f ms\n", verify_ms);
    printf("Proof size: %zu bytes\n", proof.serialize().size());

    printf("PROOF GENERATION COMPLETE.\n");
    printf("  Proof: %s\n", output_path);
    printf("  VK: %s\n", vk_path);
    printf("  Public inputs: %s\n", pi_path);

    return 0;
}
