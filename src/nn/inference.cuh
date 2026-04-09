#pragma once

#include "nn/layers.cuh"
#include "nn/quantize.cuh"
#include <vector>
#include <tuple>
#include <cstdio>

// ============================================================
// Quantized Neural Network Inference Engine
//
// Runs a complete neural network inside BN254 finite field.
// All operations are field arithmetic → ZK-provable.
//
// Supports:
// - Multi-layer perceptron (MLP) with Linear + ReLU + Softmax
// - Loads weights from flat binary format (float32)
// - Quantizes to int8/int16, converts to Fp
// - Records computation trace for witness generation
//
// Memory budget for MNIST MLP on GTX 1650:
//   Model weights (quantized to Fp): ~3.2 MB
//   Activations: negligible
//   Inference trace: ~4 MB (all intermediate values)
//   Total: ~8 MB << 3.5 GB budget
// ============================================================

namespace zkml {

using bn254::Fp;

// ============================================================
// Inference trace: records all intermediate values for witness generation
// ============================================================
struct InferenceTrace {
    std::vector<std::vector<Fp>> layer_inputs;   // Input to each layer
    std::vector<std::vector<Fp>> layer_outputs;  // Output from each layer
    std::vector<Fp> final_output;
    int num_layers;
    bool valid;

    InferenceTrace() : num_layers(0), valid(false) {}
};

// ============================================================
// Neural Network Model (quantized, in Fp)
// ============================================================
struct NNModel {
    struct LayerSpec {
        LayerType type;
        int in_size;
        int out_size;
        int seq_len;
        int hidden_size;
        int num_heads;
    };

    struct Layer {
        LayerType type;
        int in_size;
        int out_size;

        // For LINEAR layers
        std::vector<Fp> weights; // [out_size × in_size]
        std::vector<Fp> bias;    // [out_size]

        // For RELU_APPROX
        ReLUApproxParams relu_params;

        // For SOFTMAX_APPROX
        SoftmaxApproxParams softmax_params;

        // For SELF_ATTENTION
        SelfAttentionParams attention_params;
        std::vector<Fp> q_weights; // [hidden_size x hidden_size]
        std::vector<Fp> q_bias;    // [hidden_size]
        std::vector<Fp> k_weights;
        std::vector<Fp> k_bias;
        std::vector<Fp> v_weights;
        std::vector<Fp> v_bias;
        std::vector<Fp> o_weights;
        std::vector<Fp> o_bias;
    };

    std::vector<Layer> layers;
    QuantConfig weight_qconfig;
    QuantConfig input_qconfig;

    static Layer make_linear_layer(int in_size, int out_size) {
        Layer layer;
        layer.type = LayerType::LINEAR;
        layer.in_size = in_size;
        layer.out_size = out_size;
        layer.weights.resize((size_t)out_size * (size_t)in_size);
        layer.bias.resize((size_t)out_size);
        return layer;
    }

    static Layer make_relu_layer(int size) {
        Layer layer;
        layer.type = LayerType::RELU_APPROX;
        layer.in_size = size;
        layer.out_size = size;
        layer.relu_params.c0 = Fp::from_uint(64);
        layer.relu_params.c1 = Fp::from_uint(64);
        layer.relu_params.c2 = Fp::from_uint(1);
        layer.relu_params.c3 = Fp::zero();
        layer.relu_params.scale = Fp::from_uint(1);
        return layer;
    }

    static Layer make_softmax_layer(int size) {
        Layer layer;
        layer.type = LayerType::SOFTMAX_APPROX;
        layer.in_size = size;
        layer.out_size = size;
        layer.softmax_params.num_classes = size;
        layer.softmax_params.inv_scale = Fp::from_uint(1);
        return layer;
    }

    static Layer make_attention_layer(int seq_len, int hidden_size, int num_heads = 1) {
        Layer layer;
        layer.type = LayerType::SELF_ATTENTION;
        layer.in_size = seq_len * hidden_size;
        layer.out_size = seq_len * hidden_size;
        layer.attention_params.seq_len = seq_len;
        layer.attention_params.hidden_size = hidden_size;
        layer.attention_params.num_heads = num_heads > 0 ? num_heads : 1;
        layer.attention_params.inv_scale = Fp::one();
        size_t matrix_elems = (size_t)hidden_size * (size_t)hidden_size;
        layer.q_weights.resize(matrix_elems);
        layer.q_bias.resize((size_t)hidden_size);
        layer.k_weights.resize(matrix_elems);
        layer.k_bias.resize((size_t)hidden_size);
        layer.v_weights.resize(matrix_elems);
        layer.v_bias.resize((size_t)hidden_size);
        layer.o_weights.resize(matrix_elems);
        layer.o_bias.resize((size_t)hidden_size);
        return layer;
    }

    static NNModel create_from_specs(const std::vector<LayerSpec>& specs) {
        NNModel model;
        model.weight_qconfig = QuantConfig::symmetric(8, 1.0f);
        model.input_qconfig = QuantConfig::symmetric(8, 1.0f);

        for (const auto& spec : specs) {
            if (spec.in_size <= 0 || spec.out_size <= 0) {
                continue;
            }

            if (spec.type == LayerType::LINEAR) {
                model.layers.push_back(make_linear_layer(spec.in_size, spec.out_size));
            } else if (spec.type == LayerType::RELU_APPROX) {
                model.layers.push_back(make_relu_layer(spec.in_size));
            } else if (spec.type == LayerType::SOFTMAX_APPROX) {
                model.layers.push_back(make_softmax_layer(spec.in_size));
            } else if (spec.type == LayerType::SELF_ATTENTION) {
                model.layers.push_back(
                    make_attention_layer(
                        spec.seq_len,
                        spec.hidden_size,
                        spec.num_heads > 0 ? spec.num_heads : 1));
            }
        }

        return model;
    }

    static NNModel create_mlp(const std::vector<int>& dims, bool final_softmax = true) {
        std::vector<LayerSpec> specs;
        if (dims.size() < 2) {
            return create_from_specs(specs);
        }

        for (size_t i = 0; i + 1 < dims.size(); i++) {
            specs.push_back({LayerType::LINEAR, dims[i], dims[i + 1], 0, 0, 0});
            if (i + 1 < dims.size() - 1) {
                specs.push_back({LayerType::RELU_APPROX, dims[i + 1], dims[i + 1], 0, 0, 0});
            }
        }

        if (final_softmax && !dims.empty()) {
            int classes = dims.back();
            specs.push_back({LayerType::SOFTMAX_APPROX, classes, classes, 0, 0, 0});
        }

        return create_from_specs(specs);
    }

    // Create a simple MLP for MNIST: 784 → 128 → 10
    static NNModel create_mnist_mlp() {
        return create_mlp({784, 128, 10}, true);
    }

    // CIFAR-10 flattened MLP: 3072 -> 256 -> 64 -> 10
    static NNModel create_cifar_mlp() {
        return create_mlp({3072, 256, 64, 10}, true);
    }

    static NNModel create_tiny_transformer() {
        NNModel model;
        model.weight_qconfig = QuantConfig::symmetric(8, 1.0f);
        model.input_qconfig = QuantConfig::symmetric(8, 1.0f);
        model.layers.push_back(make_attention_layer(4, 8));          // 32 -> 32
        model.layers.push_back(make_relu_layer(32));
        model.layers.push_back(make_linear_layer(32, 4));            // classifier
        model.layers.push_back(make_softmax_layer(4));
        return model;
    }

    // Load weights from flat binary (float32 format)
    bool load_weights(const float* data, int total_floats) {
        int offset = 0;
        for (auto& layer : layers) {
            if (layer.type == LayerType::LINEAR) {
                int num_weights = layer.out_size * layer.in_size;
                int num_bias = layer.out_size;

                if (offset + num_weights + num_bias > total_floats) {
                    fprintf(stderr, "[NN] Error: not enough weight data\n");
                    return false;
                }

                float max_w = find_max_abs(data + offset, num_weights);
                QuantConfig wq = QuantConfig::symmetric(weight_qconfig.bits, max_w);

                layer.weights.resize(num_weights);
                quantize_array(layer.weights.data(), data + offset, num_weights, wq);
                offset += num_weights;

                float max_b = find_max_abs(data + offset, num_bias);
                QuantConfig bq = QuantConfig::symmetric(weight_qconfig.bits, max_b);
                layer.bias.resize(num_bias);
                quantize_array(layer.bias.data(), data + offset, num_bias, bq);
                offset += num_bias;
            } else if (layer.type == LayerType::SELF_ATTENTION) {
                int hidden = layer.attention_params.hidden_size;
                int matrix_elems = hidden * hidden;
                int bias_elems = hidden;
                int total = 4 * (matrix_elems + bias_elems);
                if (offset + total > total_floats) {
                    fprintf(stderr, "[NN] Error: not enough self-attention weight data\n");
                    return false;
                }

                auto quant_block = [&](std::vector<Fp>& dst, int count) {
                    float max_abs = find_max_abs(data + offset, count);
                    QuantConfig cfg = QuantConfig::symmetric(weight_qconfig.bits, max_abs);
                    dst.resize(count);
                    quantize_array(dst.data(), data + offset, count, cfg);
                    offset += count;
                };

                quant_block(layer.q_weights, matrix_elems);
                quant_block(layer.q_bias, bias_elems);
                quant_block(layer.k_weights, matrix_elems);
                quant_block(layer.k_bias, bias_elems);
                quant_block(layer.v_weights, matrix_elems);
                quant_block(layer.v_bias, bias_elems);
                quant_block(layer.o_weights, matrix_elems);
                quant_block(layer.o_bias, bias_elems);
            }
        }
        return true;
    }

    // Initialize with random weights (for testing)
    void init_random(unsigned seed = 42) {
        srand(seed);
        for (auto& layer : layers) {
            if (layer.type == LayerType::LINEAR) {
                int n_w = layer.out_size * layer.in_size;
                for (int i = 0; i < n_w; i++) {
                    int32_t v = (rand() % 7) - 3;
                    layer.weights[i] = int_to_fp(v);
                }
                for (int i = 0; i < layer.out_size; i++) {
                    layer.bias[i] = int_to_fp(0);
                }
            } else if (layer.type == LayerType::SELF_ATTENTION) {
                auto init_vec = [&](std::vector<Fp>& dst) {
                    for (size_t i = 0; i < dst.size(); i++) {
                        int32_t v = (rand() % 5) - 2;
                        dst[i] = int_to_fp(v);
                    }
                };
                init_vec(layer.q_weights);
                init_vec(layer.k_weights);
                init_vec(layer.v_weights);
                init_vec(layer.o_weights);
                for (auto* bias_vec : {&layer.q_bias, &layer.k_bias, &layer.v_bias, &layer.o_bias}) {
                    for (size_t i = 0; i < bias_vec->size(); i++) {
                        (*bias_vec)[i] = int_to_fp(0);
                    }
                }
            }
        }
    }
};

// GPU inference — runs on GPU, records trace
InferenceTrace run_inference_gpu(const NNModel& model,
                                  const Fp* h_input,
                                  int input_size);

// CPU reference inference
std::vector<Fp> run_inference_cpu(const NNModel& model,
                                    const Fp* input,
                                    int input_size);

} // namespace zkml
