// Host tests for the PCANI v2 statement (roadmap 1.1, 1.2, 2.1-2.4, 2.6).
// A small CNN (CONV2D -> RELU_EXACT -> LINEAR -> softmax) with integer weights,
// a batch of two samples, two statement variants:
//   A: exact public model, public inputs, context binding (1.1), outputs public
//   B: private model (Poseidon commitment), private inputs (commitments with
//      blinding), private outputs, in-circuit routing (route flag + class public)
// Checks: satisfiability; scores equal the host reference inference; every
// variable is pinned (single-variable perturbation rejected, roadmap 1.2);
// wrong context / tampered commitment / wrong route flag rejected; the R1CS and
// witness are exported (iden3 .r1cs/.wtns) for tests/test_underconstrained.py.
//
//   tests/host/run_host_tests.sh [export_dir]

#include "prover/statement_v2.cuh"
#include "prover/r1cs_export.cuh"
#include <cstdio>
#include <string>

using namespace zkml;
using bn254::Fp;

static int g_fail = 0;
static void expect(bool c, const char* name) {
    printf("%s %s\n", c ? "[PASS]" : "[FAIL]", name);
    if (!c) g_fail++;
}

static NNModel tiny_cnn() {
    NNModel m;
    Conv2DParams c;
    c.in_c = 1; c.in_h = 6; c.in_w = 6; c.out_c = 2; c.kh = 3; c.kw = 3; c.stride = 1; c.pad = 1;
    m.layers.push_back(NNModel::make_conv2d_layer(c));
    m.layers.push_back(NNModel::make_relu_exact_layer(c.out_size(), 24, 2));
    m.layers.push_back(NNModel::make_linear_layer(c.out_size(), 4));
    m.layers.push_back(NNModel::make_softmax_layer(4));
    auto& conv = m.layers[0];
    for (size_t i = 0; i < conv.weights.size(); i++) conv.weights[i] = i64_to_fp((int64_t)((i * 5) % 7) - 3);
    conv.bias[0] = i64_to_fp(4); conv.bias[1] = i64_to_fp(-2);
    auto& lin = m.layers[2];
    for (size_t i = 0; i < lin.weights.size(); i++) lin.weights[i] = i64_to_fp((int64_t)((i * 3) % 9) - 4);
    for (size_t i = 0; i < lin.bias.size(); i++) lin.bias[i] = i64_to_fp((int64_t)i - 1);
    return m;
}

static std::vector<int64_t> reference_scores(const NNModel& m, const std::vector<Fp>& x) {
    std::vector<Fp> h = run_conv2d_layer_host(m.layers[0], x);
    h = run_relu_exact_layer_host(m.layers[1], h);
    const auto& lin = m.layers[2];
    std::vector<int64_t> out;
    for (int i = 0; i < lin.out_size; i++) {
        Fp acc = lin.bias[(size_t)i];
        for (int j = 0; j < lin.in_size; j++) acc = acc + lin.weights[(size_t)i * lin.in_size + j] * h[(size_t)j];
        int64_t v = 0;
        fp_to_i64_checked(acc, &v);
        out.push_back(v);
    }
    return out;
}

// Single-variable perturbation check, incremental: w[i] += 1 only changes the rows
// that contain i, so only those are re-evaluated (O(nnz) instead of O(V * C)).
static bool all_pinned(const StatementV2& st, int* bad) {
    const R1CS& c = st.r1cs;
    const auto& w = st.witness.values;
    const size_t m = (size_t)c.num_constraints;
    std::vector<Fr> Aw(m, Fr::zero()), Bw(m, Fr::zero()), Cw(m, Fr::zero());
    struct Hit { int row; int mat; Fr coeff; };
    std::vector<std::vector<Hit>> by_var((size_t)c.num_variables);
    for (auto& e : c.A) { Aw[(size_t)e.row] = Aw[(size_t)e.row] + e.value * w[(size_t)e.col]; by_var[(size_t)e.col].push_back({e.row, 0, e.value}); }
    for (auto& e : c.B) { Bw[(size_t)e.row] = Bw[(size_t)e.row] + e.value * w[(size_t)e.col]; by_var[(size_t)e.col].push_back({e.row, 1, e.value}); }
    for (auto& e : c.C) { Cw[(size_t)e.row] = Cw[(size_t)e.row] + e.value * w[(size_t)e.col]; by_var[(size_t)e.col].push_back({e.row, 2, e.value}); }
    for (int i = 1; i < c.num_variables; i++) {
        const auto& hits = by_var[(size_t)i];
        if (hits.empty()) { *bad = i; return false; }        // appears in no constraint at all
        bool broken = false;
        // group hits per row
        for (size_t h = 0; h < hits.size() && !broken; h++) {
            int r = hits[h].row;
            Fr a = Aw[(size_t)r], b = Bw[(size_t)r], cc = Cw[(size_t)r];
            for (auto& x : hits) {
                if (x.row != r) continue;
                if (x.mat == 0) a = a + x.coeff; else if (x.mat == 1) b = b + x.coeff; else cc = cc + x.coeff;
            }
            if (!(a * b == cc)) broken = true;
        }
        if (!broken) { *bad = i; return false; }
    }
    return true;
}

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : ".";
    printf("== PCANI statement v2 host tests\n  (R1CS 'Constraint N failed' lines are expected negative checks)\n");
    NNModel m = tiny_cnn();
    std::vector<std::vector<Fp>> batch(2, std::vector<Fp>(36));
    for (int i = 0; i < 36; i++) {
        batch[0][(size_t)i] = i64_to_fp((i * 7) % 13 - 4);
        batch[1][(size_t)i] = i64_to_fp((i * 11) % 17 - 8);
    }
    std::vector<int64_t> ref0 = reference_scores(m, batch[0]), ref1 = reference_scores(m, batch[1]);

    // ---------------- variant A ----------------
    StatementV2Options a;
    a.model_tag = Fr::from_uint(0xC0FFEE);
    a.has_context = true;
    fr_from_hex("0x0123456789abcdef0123456789abcdef", &a.context);
    StatementV2 sa;
    std::string err;
    bool built = build_statement_v2(m, batch, a, sa, err);
    expect(built, ("A: statement builds " + err).c_str());
    if (built) {
        expect(sa.r1cs.verify_witness(sa.witness.values), "A: witness satisfies the R1CS");
        expect(sa.scores[0] == ref0 && sa.scores[1] == ref1, "A: in-circuit scores equal host reference inference (batch of 2)");
        auto pub = sa.witness.get_public();
        expect((int)pub.size() == sa.layout.num_public && sa.layout.num_public == 2 * 4 + 1 + 1 + 2 * 36,
               "A: public layout = outputs(8) + tag + context + inputs(72)");
        int bad = -1;
        bool pinned = all_pinned(sa, &bad);
        expect(pinned, "A: every variable pinned (no underconstrained wire)");
        if (!pinned) printf("  first unpinned variable: %d\n", bad);
        std::vector<Fr> w = sa.witness.values;
        w[(size_t)(1 + sa.layout.context_index)] = w[(size_t)(1 + sa.layout.context_index)] + Fr::one();
        expect(!sa.r1cs.verify_witness(w), "A: changing the context public input breaks the statement (anti-replay)");
        export_r1cs_iden3(sa.r1cs, (dir + "/stmt_v2_A.r1cs").c_str());
        export_wtns_iden3(sa.witness.values, (dir + "/stmt_v2_A.wtns").c_str());
        printf("  A: %d constraints, %d variables\n", sa.r1cs.num_constraints, sa.r1cs.num_variables);
    }

    // ---------------- variant B ----------------
    StatementV2Options bo;
    bo.model_tag = Fr::from_uint(0xC0FFEE);
    bo.private_input = true;
    bo.input_blinding = {Fr::from_uint(0x5EED1), Fr::from_uint(0x5EED2)};
    bo.private_model = true;
    bo.private_outputs = true;
    bo.route = true;
    bo.route_threshold = 5;
    StatementV2 sb;
    built = build_statement_v2(m, batch, bo, sb, err);
    expect(built, ("B: statement builds " + err).c_str());
    if (built) {
        expect(sb.r1cs.verify_witness(sb.witness.values), "B: witness satisfies the R1CS");
        auto pub = sb.witness.get_public();
        expect(sb.layout.num_public == 1 + 1 + 2 + 4, "B: public layout = tag + C_model + C_input(2) + route(2x2)");
        // expected route decisions
        bool route_ok = true;
        for (int b = 0; b < 2; b++) {
            const auto& s = b == 0 ? ref0 : ref1;
            int top = 0;
            for (int i = 1; i < 4; i++) if (s[(size_t)i] > s[(size_t)top]) top = i;
            int64_t second = INT64_MIN;
            for (int i = 0; i < 4; i++) if (i != top && s[(size_t)i] > second) second = s[(size_t)i];
            Fr flag = (s[(size_t)top] - second >= 5) ? Fr::one() : Fr::zero();
            route_ok &= pub[(size_t)(sb.layout.route_start + 2 * b)] == flag;
            route_ok &= pub[(size_t)(sb.layout.route_start + 2 * b + 1)] == Fr::from_uint((uint64_t)top);
        }
        expect(route_ok, "B: in-circuit route flag and top-1 class match the reference");
        std::vector<Fr> wts;
        for (size_t li : {(size_t)0, (size_t)2}) {
            for (auto& w : m.layers[li].weights) wts.push_back(fp_to_fr_signed(w));
            for (auto& b : m.layers[li].bias) wts.push_back(fp_to_fr_signed(b));
        }
        expect(pub[(size_t)sb.layout.model_commitment_index] == poseidon_commit_native(wts, poseidon_bn254::DOMAIN_MODEL),
               "B: C_model equals the native Poseidon commitment of the weights");
        std::vector<Fr> xin;
        for (auto& x : batch[0]) xin.push_back(fp_to_fr_signed(x));
        xin.push_back(Fr::from_uint(0x5EED1));
        expect(pub[(size_t)sb.layout.inputs_start] == poseidon_commit_native(xin, poseidon_bn254::DOMAIN_INPUT),
               "B: C_input equals the native commitment of (x || r)");
        int bad = -1;
        bool pinned = all_pinned(sb, &bad);
        expect(pinned, "B: every variable pinned (weights/inputs fixed by their commitments)");
        if (!pinned) printf("  first unpinned variable: %d\n", bad);
        std::vector<Fr> w = sb.witness.values;
        size_t fi = (size_t)(1 + sb.layout.route_start);
        w[fi] = Fr::one() - w[fi];
        expect(!sb.r1cs.verify_witness(w), "B: flipping the public route flag is rejected");
        export_r1cs_iden3(sb.r1cs, (dir + "/stmt_v2_B.r1cs").c_str());
        export_wtns_iden3(sb.witness.values, (dir + "/stmt_v2_B.wtns").c_str());
        printf("  B: %d constraints, %d variables (private model + private input + routing)\n",
               sb.r1cs.num_constraints, sb.r1cs.num_variables);
    }

    // ---------------- trained ReLU model (python/export_trained_relu_model.py) ----------------
    if (argc > 2) {
        const std::string mdir = argv[2];
        FILE* tf = fopen((mdir + "/expected_scores.txt").c_str(), "r");
        FILE* bf = fopen((mdir + "/model.int32.bin").c_str(), "rb");
        if (tf && bf) {
            int hidden = 0, frac = 0, rbits = 0, n = 0;
            if (fscanf(tf, "%d %d %d %d", &hidden, &frac, &rbits, &n) != 4) n = 0;
            NNModel tm;
            tm.layers.push_back(NNModel::make_linear_layer(64, hidden));
            tm.layers.push_back(NNModel::make_relu_exact_layer(hidden, rbits, frac));
            tm.layers.push_back(NNModel::make_linear_layer(hidden, 10));
            tm.layers.push_back(NNModel::make_softmax_layer(10));
            std::vector<int32_t> p((size_t)(64 * hidden + hidden + hidden * 10 + 10));
            bool okp = fread(p.data(), 4, p.size(), bf) == p.size();
            std::vector<int32_t> flat(p.begin(), p.end());
            okp &= tm.load_integer_weights(flat.data(), (int)flat.size());
            bool exact = okp && n > 0;
            std::vector<std::vector<Fp>> xs;
            std::vector<std::vector<int64_t>> want;
            for (int s = 0; s < n; s++) {
                std::vector<Fp> x(64);
                std::vector<int64_t> sc(10);
                for (int i = 0; i < 64; i++) { long long v = 0; if (fscanf(tf, "%lld", &v) != 1) exact = false; x[(size_t)i] = i64_to_fp(v); }
                for (int i = 0; i < 10; i++) { long long v = 0; if (fscanf(tf, "%lld", &v) != 1) exact = false; sc[(size_t)i] = v; }
                xs.push_back(x);
                want.push_back(sc);
            }
            StatementV2Options to;
            to.model_tag = Fr::from_uint(1);
            to.has_context = true;
            to.context = Fr::from_uint(42);
            to.route = true;
            to.route_threshold = 200;
            StatementV2 st;
            exact &= build_statement_v2(tm, xs, to, st, err) && st.r1cs.verify_witness(st.witness.values);
            for (int s = 0; exact && s < n; s++) exact &= st.scores[(size_t)s] == want[(size_t)s];
            expect(exact, "trained ReLU digits model: C++ statement scores == Python exact reference (batch of 8)");
            printf("  trained model statement: %d constraints for %d samples\n", st.r1cs.num_constraints, n);
        } else {
            expect(false, "trained model files readable");
        }
        if (tf) fclose(tf);
        if (bf) fclose(bf);
    }

    // ---------------- misuse ----------------
    StatementV2Options bad = bo;
    bad.input_blinding.clear();
    StatementV2 sx;
    expect(!build_statement_v2(m, batch, bad, sx, err), "private input without blinding is refused");

    printf(g_fail == 0 ? "[HOST STATEMENT V2 TESTS PASS]\n" : "[HOST STATEMENT V2 TESTS FAIL] %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
