#include "prover/witness.cuh"
#include "prover/circuit.cuh"
#include <cstdio>
#include <vector>

using namespace zkml;
using bn254::Fp;
using bn254::Fr;

static Fr fp_to_fr_local(const Fp& v) {
    uint64_t s[4] = {0,0,0,0};
    v.to_standard(s);
    Fr out;
    Fr::mont_mul_fr(out.val, s, Fr::r_squared().val);
    return out;
}

static std::vector<Fr> cv(const std::vector<Fp>& src) {
    std::vector<Fr> out(src.size());
    for (size_t i = 0; i < src.size(); ++i) out[i] = fp_to_fr_local(src[i]);
    return out;
}

static bool check_mlp() {
    NNModel m = NNModel::create_mlp({2, 3, 2}, true);
    int k = 1;
    for (auto& l : m.layers) {
        if (l.type == LayerType::LINEAR) {
            for (auto& w : l.weights) w = Fp::from_uint((k++ % 3) + 1);
            for (auto& b : l.bias) b = Fp::from_uint(1);
        }
    }
    Fp input[2] = {Fp::from_uint(2), Fp::from_uint(3)};
    InferenceTrace trace;
    Fr tag = Fr::from_uint(0x1234);
    Witness w = generate_fixed_model_witness(m, trace, input, 2, tag, true);

    CircuitBuilder b;
    int nout = circuit_public_output_size(m);
    std::vector<int> pub(nout);
    for (int i = 0; i < nout; ++i) pub[i] = b.alloc_var();
    int tagv = b.alloc_var();
    b.circuit.add_linear_constraint({{0, tag}}, {{tagv, Fr::one()}});
    std::vector<int> cur = {b.alloc_var(), b.alloc_var()};
    for (int li = 0; li < (int)m.layers.size(); ++li) {
        auto& l = m.layers[li];
        if (l.type == LayerType::LINEAR) {
            if (is_last_circuit_layer(m, li)) {
                cur = b.add_fixed_linear_layer_into(cur, pub, cv(l.weights), cv(l.bias), l.out_size, l.in_size);
            } else {
                cur = b.add_fixed_linear_layer(cur, cv(l.weights), cv(l.bias), l.out_size, l.in_size);
            }
        } else if (l.type == LayerType::RELU_APPROX) {
            cur = b.add_relu_approx(cur, l.in_size,
                                    fp_to_fr_local(l.relu_params.c0),
                                    fp_to_fr_local(l.relu_params.c1),
                                    fp_to_fr_local(l.relu_params.c2),
                                    fp_to_fr_local(l.relu_params.c3),
                                    fp_to_fr_local(l.relu_params.scale));
        }
    }
    b.finalize(nout + 1 + 2, 0);
    return b.circuit.num_variables == (int)w.values.size() && b.circuit.verify_witness(w.values);
}

static bool check_attention() {
    NNModel m = NNModel::create_tiny_transformer();
    int k = 1;
    for (auto& l : m.layers) {
        if (l.type == LayerType::SELF_ATTENTION) {
            auto fill = [&](std::vector<Fp>& v) {
                for (auto& x : v) x = Fp::from_uint((k++ % 3) + 1);
            };
            fill(l.q_weights); fill(l.q_bias); fill(l.k_weights); fill(l.k_bias);
            fill(l.v_weights); fill(l.v_bias); fill(l.o_weights); fill(l.o_bias);
        } else if (l.type == LayerType::LINEAR) {
            for (auto& w : l.weights) w = Fp::from_uint((k++ % 3) + 1);
            for (auto& b : l.bias) b = Fp::from_uint(1);
        }
    }
    std::vector<Fp> input(32);
    for (int i = 0; i < 32; ++i) input[i] = Fp::from_uint((i % 2) + 1);
    InferenceTrace trace;
    Fr tag = Fr::from_uint(0x5678);
    Witness w = generate_fixed_model_witness(m, trace, input.data(), 32, tag, true);

    CircuitBuilder b;
    int nout = circuit_public_output_size(m);
    std::vector<int> pub(nout);
    for (int i = 0; i < nout; ++i) pub[i] = b.alloc_var();
    int tagv = b.alloc_var();
    b.circuit.add_linear_constraint({{0, tag}}, {{tagv, Fr::one()}});
    std::vector<int> cur(32);
    for (auto& x : cur) x = b.alloc_var();
    for (int li = 0; li < (int)m.layers.size(); ++li) {
        auto& l = m.layers[li];
        if (l.type == LayerType::SELF_ATTENTION) {
            cur = b.add_fixed_self_attention_layer(
                cur, l.attention_params.seq_len, l.attention_params.hidden_size,
                l.attention_params.num_heads,
                cv(l.q_weights), cv(l.q_bias), cv(l.k_weights), cv(l.k_bias),
                cv(l.v_weights), cv(l.v_bias), cv(l.o_weights), cv(l.o_bias));
        } else if (l.type == LayerType::RELU_APPROX) {
            cur = b.add_relu_approx(cur, l.in_size,
                                    fp_to_fr_local(l.relu_params.c0),
                                    fp_to_fr_local(l.relu_params.c1),
                                    fp_to_fr_local(l.relu_params.c2),
                                    fp_to_fr_local(l.relu_params.c3),
                                    fp_to_fr_local(l.relu_params.scale));
        } else if (l.type == LayerType::LINEAR) {
            if (is_last_circuit_layer(m, li)) {
                cur = b.add_fixed_linear_layer_into(cur, pub, cv(l.weights), cv(l.bias), l.out_size, l.in_size);
            } else {
                cur = b.add_fixed_linear_layer(cur, cv(l.weights), cv(l.bias), l.out_size, l.in_size);
            }
        }
    }
    b.finalize(nout + 1 + 32, 0);
    return b.circuit.num_variables == (int)w.values.size() && b.circuit.verify_witness(w.values);
}

int main() {
    bool a = check_mlp();
    bool b = check_attention();
    printf("Fixed-model MLP statement: %s\n", a ? "PASS" : "FAIL");
    printf("Fixed-model attention statement: %s\n", b ? "PASS" : "FAIL");
    return (a && b) ? 0 : 1;
}
