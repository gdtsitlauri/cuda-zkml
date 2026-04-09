#include "nn/inference.cuh"
#include "common.cuh"
#include <cstring>

// ============================================================
// GPU Inference Engine Implementation
// ============================================================

namespace zkml {

using bn254::Fp;

static std::vector<Fp> run_self_attention_cpu_layer(const NNModel::Layer& layer,
                                                    const std::vector<Fp>& input) {
    const int seq_len = layer.attention_params.seq_len;
    const int hidden = layer.attention_params.hidden_size;
    const int num_heads = layer.attention_params.num_heads > 0 ? layer.attention_params.num_heads : 1;
    const int head_dim = hidden / num_heads;
    std::vector<Fp> q((size_t)seq_len * (size_t)hidden, Fp::zero());
    std::vector<Fp> k((size_t)seq_len * (size_t)hidden, Fp::zero());
    std::vector<Fp> v((size_t)seq_len * (size_t)hidden, Fp::zero());
    std::vector<Fp> scores((size_t)seq_len * (size_t)num_heads * (size_t)seq_len, Fp::zero());
    std::vector<Fp> probs((size_t)seq_len * (size_t)num_heads * (size_t)seq_len, Fp::zero());
    std::vector<Fp> context((size_t)seq_len * (size_t)hidden, Fp::zero());
    std::vector<Fp> output((size_t)seq_len * (size_t)hidden, Fp::zero());

    auto project = [&](const std::vector<Fp>& weights,
                       const std::vector<Fp>& bias,
                       std::vector<Fp>& dst) {
        for (int token = 0; token < seq_len; token++) {
            for (int out = 0; out < hidden; out++) {
                Fp acc = bias[out];
                for (int in = 0; in < hidden; in++) {
                    acc = acc + weights[(size_t)out * (size_t)hidden + (size_t)in] *
                                  input[(size_t)token * (size_t)hidden + (size_t)in];
                }
                dst[(size_t)token * (size_t)hidden + (size_t)out] = acc;
            }
        }
    };

    project(layer.q_weights, layer.q_bias, q);
    project(layer.k_weights, layer.k_bias, k);
    project(layer.v_weights, layer.v_bias, v);

    for (int token = 0; token < seq_len; token++) {
        for (int head = 0; head < num_heads; head++) {
            std::vector<Fp> exp_scores((size_t)seq_len, Fp::zero());
            Fp sum = Fp::zero();

            for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                Fp score = Fp::zero();
                for (int d = 0; d < head_dim; d++) {
                    int idx = head * head_dim + d;
                    score = score + q[(size_t)token * (size_t)hidden + (size_t)idx] *
                                    k[(size_t)key_idx * (size_t)hidden + (size_t)idx];
                }
                score = score * layer.attention_params.inv_scale;
                scores[((size_t)token * (size_t)num_heads + (size_t)head) * (size_t)seq_len +
                       (size_t)key_idx] = score;

                Fp x2 = score * score;
                Fp x3 = x2 * score;
                Fp exp_approx = Fp::one() + score + x2 + x3;
                exp_scores[(size_t)key_idx] = exp_approx;
                sum = sum + exp_approx;
            }

            Fp sum_inv = sum.inv();
            for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                Fp prob = exp_scores[(size_t)key_idx] * sum_inv;
                probs[((size_t)token * (size_t)num_heads + (size_t)head) * (size_t)seq_len +
                      (size_t)key_idx] = prob;
            }
        }
    }

    for (int token = 0; token < seq_len; token++) {
        for (int head = 0; head < num_heads; head++) {
            for (int d = 0; d < head_dim; d++) {
                int idx = head * head_dim + d;
                Fp acc = Fp::zero();
                for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                    acc = acc + probs[((size_t)token * (size_t)num_heads + (size_t)head) * (size_t)seq_len +
                                      (size_t)key_idx] *
                                v[(size_t)key_idx * (size_t)hidden + (size_t)idx];
                }
                context[(size_t)token * (size_t)hidden + (size_t)idx] = acc;
            }
        }
    }

    for (int token = 0; token < seq_len; token++) {
        for (int out = 0; out < hidden; out++) {
            Fp acc = layer.o_bias[out];
            for (int in = 0; in < hidden; in++) {
                acc = acc + layer.o_weights[(size_t)out * (size_t)hidden + (size_t)in] *
                              context[(size_t)token * (size_t)hidden + (size_t)in];
            }
            // Residual connection keeps the transformer block shape-preserving.
            acc = acc + input[(size_t)token * (size_t)hidden + (size_t)out];
            output[(size_t)token * (size_t)hidden + (size_t)out] = acc;
        }
    }

    return output;
}

// Run inference on GPU, record trace for witness generation
InferenceTrace run_inference_gpu(const NNModel& model,
                                  const Fp* h_input,
                                  int input_size) {
    InferenceTrace trace;
    trace.num_layers = (int)model.layers.size();

    print_gpu_memory("Inference start");

    // Find max layer sizes for allocation
    int max_size = input_size;
    for (const auto& layer : model.layers) {
        max_size = max(max_size, max(layer.in_size, layer.out_size));
    }

    // Allocate working buffers
    Fp* d_input;
    Fp* d_output;
    CUDA_CHECK(cudaMalloc(&d_input, max_size * sizeof(Fp)));
    CUDA_CHECK(cudaMalloc(&d_output, max_size * sizeof(Fp)));

    // Upload input
    CUDA_CHECK(cudaMemcpy(d_input, h_input, input_size * sizeof(Fp),
                           cudaMemcpyHostToDevice));

    // Record input in trace
    trace.layer_inputs.push_back(std::vector<Fp>(h_input, h_input + input_size));

    Fp* current_input = d_input;
    Fp* current_output = d_output;

    for (int l = 0; l < (int)model.layers.size(); l++) {
        const auto& layer = model.layers[l];

        switch (layer.type) {
            case LayerType::LINEAR: {
                // Upload weights and bias
                Fp* d_W;
                Fp* d_bias;
                int w_size = layer.out_size * layer.in_size;
                CUDA_CHECK(cudaMalloc(&d_W, w_size * sizeof(Fp)));
                CUDA_CHECK(cudaMalloc(&d_bias, layer.out_size * sizeof(Fp)));
                CUDA_CHECK(cudaMemcpy(d_W, layer.weights.data(),
                                       w_size * sizeof(Fp), cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_bias, layer.bias.data(),
                                       layer.out_size * sizeof(Fp), cudaMemcpyHostToDevice));

                // y = Wx + b
                fp_matmul_gpu(current_output, d_W, current_input, d_bias,
                              layer.out_size, layer.in_size);
                CUDA_CHECK(cudaDeviceSynchronize());

                CUDA_CHECK(cudaFree(d_W));
                CUDA_CHECK(cudaFree(d_bias));
                break;
            }

            case LayerType::RELU_APPROX: {
                fp_relu_approx_gpu(current_output, current_input,
                                    layer.relu_params, layer.in_size);
                CUDA_CHECK(cudaDeviceSynchronize());
                break;
            }

            case LayerType::SOFTMAX_APPROX: {
                fp_softmax_approx_gpu(current_output, current_input,
                                       layer.softmax_params, layer.in_size);
                CUDA_CHECK(cudaDeviceSynchronize());
                break;
            }

            case LayerType::SELF_ATTENTION: {
                std::vector<Fp> host_input(layer.in_size);
                CUDA_CHECK(cudaMemcpy(host_input.data(), current_input,
                                       (size_t)layer.in_size * sizeof(Fp),
                                       cudaMemcpyDeviceToHost));
                std::vector<Fp> host_output = run_self_attention_cpu_layer(layer, host_input);
                CUDA_CHECK(cudaMemcpy(current_output, host_output.data(),
                                       (size_t)layer.out_size * sizeof(Fp),
                                       cudaMemcpyHostToDevice));
                break;
            }
        }

        // Record output in trace
        std::vector<Fp> h_out(layer.out_size);
        CUDA_CHECK(cudaMemcpy(h_out.data(), current_output,
                               layer.out_size * sizeof(Fp), cudaMemcpyDeviceToHost));
        trace.layer_outputs.push_back(h_out);

        // Swap buffers for next layer
        if (l < (int)model.layers.size() - 1) {
            // Record next layer's input
            trace.layer_inputs.push_back(h_out);
            // Copy output to input for next layer
            CUDA_CHECK(cudaMemcpy(current_input, current_output,
                                   layer.out_size * sizeof(Fp), cudaMemcpyDeviceToDevice));
        }
    }

    // Copy final output
    int out_size = model.layers.back().out_size;
    trace.final_output.resize(out_size);
    CUDA_CHECK(cudaMemcpy(trace.final_output.data(), current_output,
                           out_size * sizeof(Fp), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    trace.valid = true;
    print_gpu_memory("Inference end");

    return trace;
}

// CPU reference inference (for correctness comparison)
std::vector<Fp> run_inference_cpu(const NNModel& model,
                                    const Fp* input,
                                    int input_size) {
    std::vector<Fp> current(input, input + input_size);

    for (const auto& layer : model.layers) {
        std::vector<Fp> output(layer.out_size, Fp::zero());

        switch (layer.type) {
            case LayerType::LINEAR: {
                for (int i = 0; i < layer.out_size; i++) {
                    Fp acc = Fp::zero();
                    for (int j = 0; j < layer.in_size; j++) {
                        acc = acc + layer.weights[i * layer.in_size + j] * current[j];
                    }
                    if (!layer.bias.empty()) {
                        acc = acc + layer.bias[i];
                    }
                    output[i] = acc;
                }
                break;
            }
            case LayerType::RELU_APPROX: {
                for (int i = 0; i < layer.in_size; i++) {
                    Fp x = current[i];
                    Fp result = layer.relu_params.c3;
                    result = result * x + layer.relu_params.c2;
                    result = result * x + layer.relu_params.c1;
                    result = result * x + layer.relu_params.c0;
                    result = result * layer.relu_params.scale;
                    output[i] = result;
                }
                break;
            }
            case LayerType::SOFTMAX_APPROX: {
                // Simple pass-through for CPU reference
                for (int i = 0; i < layer.in_size; i++) {
                    output[i] = current[i];
                }
                break;
            }

            case LayerType::SELF_ATTENTION: {
                output = run_self_attention_cpu_layer(layer, current);
                break;
            }
        }

        current = output;
    }

    return current;
}

} // namespace zkml
