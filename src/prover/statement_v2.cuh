#pragma once

// ============================================================
// PCANI statement v2 (roadmap 1.1, 2.1-2.4, 2.6)
//
// One R1CS statement + witness, built in a single traced pass, for B samples of
// a quantized model with layers LINEAR, CONV2D, RELU_EXACT, RELU_APPROX
// (trailing SOFTMAX_APPROX stays outside the statement, as in v1).
//
// Public input layout (in this order; absent groups take no slot):
//   [outputs]        B x n_out pre-softmax scores          (unless private_outputs)
//   [model_tag]      1, fixed by a constant constraint
//   [context]        1, bound by ctx*ctx = ctx_sq           (has_context; anti-replay)
//   [C_model]        1, Poseidon commitment to all weights  (private_model)
//   per sample b:    n_in public inputs, or C_input_b = Commit(x_b || r_b) (private_input)
//   per sample b:    route flag [top1 - top2 >= threshold], top1 class   (route)
// With fixed (public) model, weights are circuit coefficients (v1 binding).
// With private_model, weights are private witness values tied to C_model and
// one proving key serves every model of the same architecture.
// ============================================================

#include "prover/gadgets.cuh"
#include "prover/witness.cuh"
#include "nn/inference.cuh"
#include <sstream>
#include <string>
#include <vector>

namespace zkml {

struct StatementV2Options {
    Fr model_tag = Fr::zero();
    bool has_context = false;
    Fr context = Fr::zero();
    bool private_input = false;
    std::vector<Fr> input_blinding;      // one per sample, required with private_input
    bool private_model = false;
    bool private_outputs = false;        // requires route (otherwise nothing about the result is public)
    bool route = false;
    int64_t route_threshold = 0;         // flag = [top1 - top2 >= threshold]
    int compare_bits = 48;               // range of score differences for routing comparisons
};

struct StatementV2Layout {
    int batch = 0, n_in = 0, n_out = 0;
    int outputs_start = -1, model_tag_index = -1, context_index = -1, model_commitment_index = -1;
    int inputs_start = -1, input_slot_size = 0, route_start = -1;
    int num_public = 0;
};

struct StatementV2 {
    R1CS r1cs;
    Witness witness;
    StatementV2Layout layout;
    std::vector<std::vector<int64_t>> scores;   // per sample, final pre-softmax scores (for checks)
};

inline Fr fp_to_fr_signed(const bn254::Fp& v) {
    int64_t x = 0;
    if (!fp_to_i64_checked(v, &x)) x = 0;
    return fr_from_i64(x);
}

namespace detail {

// select: returns b + f * (a - b) as a fresh variable (one constraint)
inline int add_select(TracedBuilder& tb, int f, const LC& a, const LC& b) {
    LC diff = a;
    for (auto& [v, c] : b) diff.push_back({v, -c});
    Fr p = tb.val(f) * tb.eval(diff);
    int pv = tb.var(p);
    tb.constrain({{f, Fr::one()}}, diff, {{pv, Fr::one()}});
    LC out = b;
    out.push_back({pv, Fr::one()});
    int ov = tb.var(tb.eval(out));
    tb.linear(out, ov);
    return ov;
}

}  // namespace detail

inline bool build_statement_v2(const NNModel& model, const std::vector<std::vector<bn254::Fp>>& inputs,
                               const StatementV2Options& opt, StatementV2& out, std::string& err) {
    const int B = (int)inputs.size();
    if (B <= 0) { err = "no input samples"; return false; }
    const int n_in = (int)inputs[0].size();
    for (auto& x : inputs) if ((int)x.size() != n_in) { err = "batch inputs differ in size"; return false; }
    const int n_out = circuit_public_output_size(model);
    if (n_out <= 0) { err = "cannot determine output size"; return false; }
    if (opt.private_input && (int)opt.input_blinding.size() != B) { err = "private_input needs one blinding value per sample"; return false; }
    if (opt.private_outputs && !opt.route) { err = "private_outputs requires route"; return false; }
    if (opt.route && n_out < 2) { err = "route needs at least two output classes"; return false; }
    int last_circuit_layer = -1;
    for (int i = 0; i < (int)model.layers.size(); i++) {
        LayerType t = model.layers[(size_t)i].type;
        if (t == LayerType::SELF_ATTENTION) { err = "SELF_ATTENTION is not supported by statement v2"; return false; }
        if (t != LayerType::SOFTMAX_APPROX) last_circuit_layer = i;
        else if (i != (int)model.layers.size() - 1) { err = "SOFTMAX_APPROX only allowed as the last layer"; return false; }
    }

    TracedBuilder tb;
    StatementV2Layout& L = out.layout;
    L = StatementV2Layout();
    L.batch = B; L.n_in = n_in; L.n_out = n_out;

    // ---- public variables first ----
    int idx = 0;
    std::vector<int> out_pub;
    if (!opt.private_outputs) {
        L.outputs_start = idx;
        for (int i = 0; i < B * n_out; i++) { out_pub.push_back(tb.var_unset()); idx++; }
    }
    L.model_tag_index = idx++;
    int tag_var = tb.var(opt.model_tag);
    int ctx_var = -1;
    if (opt.has_context) { L.context_index = idx++; ctx_var = tb.var(opt.context); }
    int cmodel_var = -1;
    if (opt.private_model) { L.model_commitment_index = idx++; cmodel_var = tb.var_unset(); }
    L.inputs_start = idx;
    L.input_slot_size = opt.private_input ? 1 : n_in;
    std::vector<std::vector<int>> in_vars((size_t)B);
    std::vector<int> cin_vars((size_t)B, -1);
    for (int b = 0; b < B; b++) {
        if (opt.private_input) { cin_vars[(size_t)b] = tb.var_unset(); idx++; }
        else for (int i = 0; i < n_in; i++) { in_vars[(size_t)b].push_back(tb.var(fp_to_fr_signed(inputs[(size_t)b][(size_t)i]))); idx++; }
    }
    std::vector<int> route_flag((size_t)B, -1), route_class((size_t)B, -1);
    if (opt.route) {
        L.route_start = idx;
        for (int b = 0; b < B; b++) { route_flag[(size_t)b] = tb.var_unset(); route_class[(size_t)b] = tb.var_unset(); idx += 2; }
    }
    L.num_public = idx;

    // ---- statement bindings ----
    tb.cb.circuit.add_linear_constraint({{0, opt.model_tag}}, {{tag_var, Fr::one()}});
    if (opt.has_context) add_context_binding(tb, ctx_var);

    // ---- private model weights ----
    std::vector<std::vector<int>> w_vars(model.layers.size()), b_vars(model.layers.size());
    if (opt.private_model) {
        std::vector<int> all;
        for (size_t li = 0; li < model.layers.size(); li++) {
            const auto& layer = model.layers[li];
            if (layer.type != LayerType::LINEAR && layer.type != LayerType::CONV2D) continue;
            for (auto& wv : layer.weights) { int v = tb.var(fp_to_fr_signed(wv)); w_vars[li].push_back(v); all.push_back(v); }
            for (auto& bv : layer.bias) { int v = tb.var(fp_to_fr_signed(bv)); b_vars[li].push_back(v); all.push_back(v); }
        }
        add_poseidon_commit(tb, all, poseidon_bn254::DOMAIN_MODEL, cmodel_var);
    }

    // ---- private inputs + commitments ----
    if (opt.private_input) {
        for (int b = 0; b < B; b++) {
            for (int i = 0; i < n_in; i++) in_vars[(size_t)b].push_back(tb.var(fp_to_fr_signed(inputs[(size_t)b][(size_t)i])));
            std::vector<int> cv = in_vars[(size_t)b];
            cv.push_back(tb.var(opt.input_blinding[(size_t)b]));
            add_poseidon_commit(tb, cv, poseidon_bn254::DOMAIN_INPUT, cin_vars[(size_t)b]);
        }
    }

    // ---- per-sample model evaluation ----
    out.scores.assign((size_t)B, {});
    for (int b = 0; b < B; b++) {
        std::vector<int> cur = in_vars[(size_t)b];
        for (int li = 0; li <= last_circuit_layer; li++) {
            const auto& layer = model.layers[(size_t)li];
            if ((int)cur.size() != layer.in_size) { err = "layer size mismatch at layer " + std::to_string(li); return false; }
            switch (layer.type) {
                case LayerType::LINEAR:
                    if (opt.private_model) cur = add_private_linear(tb, cur, w_vars[(size_t)li], b_vars[(size_t)li], layer.out_size, layer.in_size);
                    else {
                        std::vector<Fr> w(layer.weights.size()), bias(layer.bias.size());
                        for (size_t i = 0; i < w.size(); i++) w[i] = fp_to_fr_signed(layer.weights[i]);
                        for (size_t i = 0; i < bias.size(); i++) bias[i] = fp_to_fr_signed(layer.bias[i]);
                        cur = add_fixed_linear(tb, cur, w, bias, layer.out_size, layer.in_size);
                    }
                    break;
                case LayerType::CONV2D:
                    if (opt.private_model) cur = add_private_conv2d(tb, cur, layer.conv, w_vars[(size_t)li], b_vars[(size_t)li]);
                    else {
                        std::vector<Fr> w(layer.weights.size()), bias(layer.bias.size());
                        for (size_t i = 0; i < w.size(); i++) w[i] = fp_to_fr_signed(layer.weights[i]);
                        for (size_t i = 0; i < bias.size(); i++) bias[i] = fp_to_fr_signed(layer.bias[i]);
                        cur = add_fixed_conv2d(tb, cur, layer.conv, w, bias);
                    }
                    break;
                case LayerType::RELU_EXACT:
                    for (auto& v : cur) {
                        int y = add_relu_exact(tb, v, layer.relu_exact.bits);
                        v = add_rescale(tb, y, layer.relu_exact.shift, layer.relu_exact.bits);
                    }
                    break;
                case LayerType::RELU_APPROX: {
                    Fr c0 = fp_to_fr_signed(layer.relu_params.c0), c1 = fp_to_fr_signed(layer.relu_params.c1);
                    Fr c2 = fp_to_fr_signed(layer.relu_params.c2), c3 = fp_to_fr_signed(layer.relu_params.c3);
                    Fr sc = fp_to_fr_signed(layer.relu_params.scale);
                    for (auto& v : cur) {
                        int x2 = tb.var(tb.val(v) * tb.val(v)); tb.mul(v, v, x2);
                        int x3 = tb.var(tb.val(x2) * tb.val(v)); tb.mul(x2, v, x3);
                        LC lc = {{0, sc * c0}, {v, sc * c1}, {x2, sc * c2}, {x3, sc * c3}};
                        int y = tb.var(tb.eval(lc));
                        tb.linear(lc, y);
                        v = y;
                    }
                    break;
                }
                default:
                    break;
            }
        }
        if ((int)cur.size() != n_out) { err = "final layer size mismatch"; return false; }
        for (int i = 0; i < n_out; i++) {
            int64_t s = 0;
            fr_to_i64(tb.val(cur[(size_t)i]), &s);
            out.scores[(size_t)b].push_back(s);
            if (!opt.private_outputs) {
                int pv = out_pub[(size_t)b * n_out + i];
                tb.set(pv, tb.val(cur[(size_t)i]));
                tb.constrain_zero({{pv, Fr::one()}, {cur[(size_t)i], -Fr::one()}});
            }
        }

        // ---- in-circuit routing on (possibly private) scores ----
        if (opt.route) {
            const int K = opt.compare_bits;
            // first two scores
            int f = add_geq_flag(tb, cur[1], cur[0], 1, K);                 // s1 > s0
            int m1 = detail::add_select(tb, f, {{cur[1], Fr::one()}}, {{cur[0], Fr::one()}});
            int m2 = detail::add_select(tb, f, {{cur[0], Fr::one()}}, {{cur[1], Fr::one()}});
            int i1 = detail::add_select(tb, f, {{0, Fr::one()}}, {});
            for (int j = 2; j < n_out; j++) {
                int x = cur[(size_t)j];
                int f1 = add_geq_flag(tb, x, m1, 1, K);                     // x > m1
                int f2 = add_geq_flag(tb, x, m2, 1, K);                     // x > m2
                int t = detail::add_select(tb, f2, {{x, Fr::one()}}, {{m2, Fr::one()}});
                int nm2 = detail::add_select(tb, f1, {{m1, Fr::one()}}, {{t, Fr::one()}});
                int nm1 = detail::add_select(tb, f1, {{x, Fr::one()}}, {{m1, Fr::one()}});
                int ni1 = detail::add_select(tb, f1, {{0, Fr::from_uint((uint64_t)j)}}, {{i1, Fr::one()}});
                m1 = nm1; m2 = nm2; i1 = ni1;
            }
            int flag = add_geq_flag(tb, m1, m2, opt.route_threshold, K);
            tb.set(route_flag[(size_t)b], tb.val(flag));
            tb.constrain_zero({{route_flag[(size_t)b], Fr::one()}, {flag, -Fr::one()}});
            tb.set(route_class[(size_t)b], tb.val(i1));
            tb.constrain_zero({{route_class[(size_t)b], Fr::one()}, {i1, -Fr::one()}});
        }
    }

    const int num_private_inputs = opt.private_input ? B * n_in : 0;
    tb.cb.finalize(L.num_public, num_private_inputs);
    tb.w.resize((size_t)tb.cb.circuit.num_variables, Fr::zero());
    out.r1cs = tb.cb.circuit;
    out.witness.values = tb.w;
    out.witness.num_public = L.num_public;
    out.witness.num_private = num_private_inputs;
    return true;
}

inline std::string statement_v2_meta_json(const StatementV2& st, const StatementV2Options& opt, uint64_t model_tag64) {
    const StatementV2Layout& L = st.layout;
    std::ostringstream o;
    char tag[32];
    snprintf(tag, sizeof(tag), "0x%016llx", (unsigned long long)model_tag64);
    o << "{\n"
      << "  \"statement_version\": 2,\n"
      << "  \"statement_mode\": \"pcani-v2\",\n"
      << "  \"model_binding\": \"" << (opt.private_model ? "poseidon-commitment-private-weights" : "fixed-quantized-parameters-in-r1cs") << "\",\n"
      << "  \"model_tag64\": \"" << tag << "\",\n"
      << "  \"input_visibility\": \"" << (opt.private_input ? "poseidon-commitment" : "public") << "\",\n"
      << "  \"output_visibility\": \"" << (opt.private_outputs ? "private" : "public") << "\",\n"
      << "  \"batch\": " << L.batch << ",\n"
      << "  \"input_size\": " << L.n_in << ",\n"
      << "  \"output_size\": " << L.n_out << ",\n"
      << "  \"outputs_start_index\": " << L.outputs_start << ",\n"
      << "  \"model_tag_public_index\": " << L.model_tag_index << ",\n"
      << "  \"context_public_index\": " << L.context_index << ",\n"
      << "  \"context\": \"" << (opt.has_context ? fr_to_hex(opt.context) : std::string("")) << "\",\n"
      << "  \"model_commitment_index\": " << L.model_commitment_index << ",\n"
      << "  \"inputs_start_index\": " << L.inputs_start << ",\n"
      << "  \"input_slot_size\": " << L.input_slot_size << ",\n"
      << "  \"route_start_index\": " << L.route_start << ",\n"
      << "  \"route_threshold\": " << opt.route_threshold << ",\n"
      << "  \"num_public\": " << L.num_public << ",\n"
      << "  \"r1cs_constraints\": " << st.r1cs.num_constraints << ",\n"
      << "  \"r1cs_variables\": " << st.r1cs.num_variables << "\n"
      << "}\n";
    return o.str();
}

}  // namespace zkml
