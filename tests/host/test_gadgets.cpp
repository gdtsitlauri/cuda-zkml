// Host unit tests for the PCANI v2 R1CS gadgets (roadmap 1.1, 1.2, 2.1-2.4).
// Build without CUDA:   tests/host/run_host_tests.sh   (any C++17 compiler)
// Build with CUDA:      ctest (target test_host_gadgets, compiled as CUDA host code)
//
// For every gadget: (1) the witness satisfies the R1CS and equals an independent
// reference; (2) underconstraint check (roadmap 1.2): every variable that is not
// a free input is perturbed by +1 and the R1CS must reject it; (3) out-of-range
// or forged values are rejected.

#include "prover/gadgets.cuh"
#include <cstdio>
#include <set>
#include <string>

using namespace zkml;

static int g_fail = 0;
static void expect(bool c, const char* name) {
    printf("%s %s\n", c ? "[PASS]" : "[FAIL]", name);
    if (!c) g_fail++;
}

static Fr dec(const char* s) {
    Fr x = Fr::zero(), ten = Fr::from_uint(10);
    for (const char* p = s; *p; ++p) x = x * ten + Fr::from_uint((uint64_t)(*p - '0'));
    return x;
}

static bool satisfied(TracedBuilder& tb) {
    tb.cb.finalize(0, 0);
    return tb.r1cs().verify_witness(tb.w);
}

// Perturb every variable not in `free_vars`; each perturbation must break the R1CS.
static bool all_pinned(TracedBuilder& tb, const std::set<int>& free_vars, int* first_bad = nullptr) {
    tb.cb.finalize(0, 0);
    for (int i = 1; i < (int)tb.w.size(); i++) {
        if (free_vars.count(i)) continue;
        std::vector<Fr> w = tb.w;
        w[(size_t)i] = w[(size_t)i] + Fr::one();
        if (tb.r1cs().verify_witness(w)) {
            if (first_bad) *first_bad = i;
            return false;
        }
    }
    return true;
}

static void quiet_verify_note() { printf("  (R1CS 'Constraint N failed' lines below are expected negative checks)\n"); }

int main() {
    printf("== PCANI v2 gadget host tests\n");
    quiet_verify_note();

    // ---- exact ReLU ----
    {
        const int64_t zs[] = {-5, 0, 7, -(1 << 15), (1 << 15) - 1};
        bool ok = true, pinned = true;
        for (int64_t z : zs) {
            TracedBuilder tb;
            int zv = tb.var(fr_from_i64(z));
            int y = add_relu_exact(tb, zv, 16);
            int64_t yv = 0;
            ok &= fr_to_i64(tb.val(y), &yv) && yv == (z > 0 ? z : 0) && satisfied(tb);
            pinned &= all_pinned(tb, {zv});
        }
        expect(ok, "RELU_EXACT: y = max(z,0) for range endpoints, R1CS satisfied");
        expect(pinned, "RELU_EXACT: every non-input variable is uniquely pinned (+1 perturbation rejected)");
        TracedBuilder tb;
        int zv = tb.var(fr_from_i64(1 << 15));            // out of the 16-bit signed range
        add_relu_exact(tb, zv, 16);
        expect(!satisfied(tb), "RELU_EXACT: out-of-range input cannot be proven");
        TracedBuilder tb2;
        int zv2 = tb2.var(fr_from_i64(-9));
        int y2 = add_relu_exact(tb2, zv2, 16);
        tb2.set(y2, fr_from_i64(-9));                      // claim relu(-9) = -9
        expect(!satisfied(tb2), "RELU_EXACT: forged output (relu(-9) = -9) rejected");
    }

    // ---- rescale (floor division by 2^s) ----
    {
        struct { int64_t y; int s; int64_t q; } cases[] = {{1000, 4, 62}, {-7, 2, -2}, {0, 3, 0}, {255, 8, 0}};
        bool ok = true, pinned = true;
        for (auto& c : cases) {
            TracedBuilder tb;
            int yv = tb.var(fr_from_i64(c.y));
            int q = add_rescale(tb, yv, c.s, 24);
            int64_t qv = 0;
            ok &= fr_to_i64(tb.val(q), &qv) && qv == c.q && satisfied(tb);
            pinned &= all_pinned(tb, {yv});
        }
        expect(ok, "RESCALE: q = floor(y / 2^s) incl. negatives, R1CS satisfied");
        expect(pinned, "RESCALE: quotient and remainder uniquely pinned");
    }

    // ---- comparison flag ----
    {
        bool ok = true, pinned = true;
        int64_t cases[][3] = {{10, 3, 7}, {10, 3, 8}, {-4, -4, 0}, {2, 9, -7}};
        for (auto& c : cases) {
            TracedBuilder tb;
            int a = tb.var(fr_from_i64(c[0])), b = tb.var(fr_from_i64(c[1]));
            int f = add_geq_flag(tb, a, b, c[2], 20);
            bool exp = (c[0] - c[1] - c[2]) >= 0;
            ok &= (tb.val(f) == (exp ? Fr::one() : Fr::zero())) && satisfied(tb);
            pinned &= all_pinned(tb, {a, b});
        }
        expect(ok, "GEQ_FLAG: f = [a - b >= t], R1CS satisfied");
        expect(pinned, "GEQ_FLAG: flag and bits uniquely pinned");
    }

    // ---- context binding ----
    {
        TracedBuilder tb;
        Fr ctx;
        fr_from_hex("0x1c0ffee0ddba11", &ctx);
        int cv = tb.var(ctx);
        add_context_binding(tb, cv);
        expect(satisfied(tb), "CONTEXT: ctx * ctx = ctx_sq satisfied");
        expect(all_pinned(tb, {cv}), "CONTEXT: ctx_sq pinned by ctx");
        int nconstraints_with_ctx = 0;
        for (auto& e : tb.r1cs().A) if (e.col == cv) nconstraints_with_ctx++;
        expect(nconstraints_with_ctx > 0, "CONTEXT: context variable appears in a constraint (non-zero IC term)");
    }

    // ---- sparse fixed linear + conv2d ----
    {
        Conv2DShape s{1, 4, 4, 2, 3, 3, 1, 1};
        std::vector<Fr> w(s.weight_count()), b(2);
        for (size_t i = 0; i < w.size(); i++) w[i] = fr_from_i64((int64_t)(i % 5) - 2);   // includes zeros
        b[0] = fr_from_i64(3); b[1] = fr_from_i64(-1);
        TracedBuilder tb;
        std::vector<int> x(16);
        std::vector<int64_t> xi(16);
        std::set<int> inputs;
        for (int i = 0; i < 16; i++) { xi[(size_t)i] = (i * 7) % 11 - 5; x[(size_t)i] = tb.var(fr_from_i64(xi[(size_t)i])); inputs.insert(x[(size_t)i]); }
        std::vector<int> y = add_fixed_conv2d(tb, x, s, w, b);
        bool ok = satisfied(tb) && (int)y.size() == s.out_size();
        for (int oc = 0; oc < 2 && ok; oc++)
            for (int oy = 0; oy < 4; oy++)
                for (int ox = 0; ox < 4; ox++) {
                    int64_t acc = oc == 0 ? 3 : -1;
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++) {
                            int iy = oy + ky - 1, ix = ox + kx - 1;
                            if (iy < 0 || ix < 0 || iy >= 4 || ix >= 4) continue;
                            acc += ((int64_t)(((oc * 1 + 0) * 3 + ky) * 3 + kx) % 5 - 2) * xi[(size_t)(iy * 4 + ix)];
                        }
                    int64_t got = 0;
                    ok &= fr_to_i64(tb.val(y[(size_t)((oc * 4 + oy) * 4 + ox)]), &got) && got == acc;
                }
        expect(ok, "CONV2D: sparse conv equals direct convolution (padding, zero taps skipped)");
        expect(all_pinned(tb, inputs), "CONV2D: outputs uniquely pinned");
        size_t terms = tb.r1cs().A.size();
        expect(terms < (size_t)s.out_size() * (s.in_size() + 1), "CONV2D: R1CS terms fewer than dense unrolling");
    }

    // ---- Poseidon ----
    {
        expect(poseidon_hash2_native(Fr::from_uint(1), Fr::from_uint(2)) ==
                   dec("7853200120776062878684798364095072458815029376092732009249414926327459813530"),
               "POSEIDON: native hash2(1,2) equals circomlib reference");
        TracedBuilder tb;
        int a = tb.var(Fr::from_uint(1)), b = tb.var(Fr::from_uint(2));
        int h = add_poseidon_hash2(tb, {{a, Fr::one()}}, {{b, Fr::one()}});
        expect(tb.val(h) == poseidon_hash2_native(Fr::from_uint(1), Fr::from_uint(2)) && satisfied(tb),
               "POSEIDON: in-circuit hash2 equals native, R1CS satisfied");
        printf("  Poseidon hash2 constraints: %d\n", tb.r1cs().num_constraints);
        int bad = -1;
        bool pinned = all_pinned(tb, {a, b}, &bad);
        expect(pinned, "POSEIDON: every intermediate uniquely pinned");
        if (!pinned) printf("  first unpinned var: %d\n", bad);

        std::vector<Fr> v = {Fr::from_uint(1), Fr::from_uint(2), Fr::from_uint(3), Fr::from_uint(4), Fr::from_uint(5)};
        expect(poseidon_commit_native(v, poseidon_bn254::DOMAIN_MODEL) ==
                   dec("3508929059074620887801063817879916348866360521201526977122764380756820441871"),
               "POSEIDON: native model commitment equals Python reference");
        std::vector<Fr> v2 = {fr_from_i64(7), fr_from_i64(-3)};
        expect(poseidon_commit_native(v2, poseidon_bn254::DOMAIN_INPUT) ==
                   dec("3582585096224963999013983200125775183577038371087651758574018517215136713633"),
               "POSEIDON: native input commitment (negative entries) equals Python reference");
        TracedBuilder tc;
        std::vector<int> vars;
        for (auto& x : v) vars.push_back(tc.var(x));
        int c = add_poseidon_commit(tc, vars, poseidon_bn254::DOMAIN_MODEL);
        expect(tc.val(c) == poseidon_commit_native(v, poseidon_bn254::DOMAIN_MODEL) && satisfied(tc),
               "POSEIDON: in-circuit commitment equals native, R1CS satisfied");
        std::vector<Fr> w2 = tc.w;
        w2[(size_t)vars[2]] = w2[(size_t)vars[2]] + Fr::one();   // change a committed value, keep the commitment
        tc.cb.finalize(0, 0);
        expect(!tc.r1cs().verify_witness(w2), "POSEIDON: changing a committed value breaks the commitment");
    }

    // ---- private linear layer ----
    {
        TracedBuilder tb;
        std::vector<int> x = {tb.var(fr_from_i64(2)), tb.var(fr_from_i64(-3))};
        std::vector<int> wv = {tb.var(fr_from_i64(4)), tb.var(fr_from_i64(1)), tb.var(fr_from_i64(-2)), tb.var(fr_from_i64(5))};
        std::vector<int> bv = {tb.var(fr_from_i64(1)), tb.var(fr_from_i64(0))};
        std::vector<int> y = add_private_linear(tb, x, wv, bv, 2, 2);
        int64_t y0 = 0, y1 = 0;
        fr_to_i64(tb.val(y[0]), &y0); fr_to_i64(tb.val(y[1]), &y1);
        expect(y0 == 4 * 2 + 1 * -3 + 1 && y1 == -2 * 2 + 5 * -3 && satisfied(tb), "PRIVATE_LINEAR: y = Wx + b");
        std::set<int> free_vars(x.begin(), x.end());
        free_vars.insert(wv.begin(), wv.end());
        free_vars.insert(bv.begin(), bv.end());
        expect(all_pinned(tb, free_vars), "PRIVATE_LINEAR: products and outputs pinned");
    }

    printf(g_fail == 0 ? "[HOST GADGET TESTS PASS]\n" : "[HOST GADGET TESTS FAIL] %d\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
