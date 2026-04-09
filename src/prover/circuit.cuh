#pragma once

#include "field/fp.cuh"
#include <vector>

// ============================================================
// R1CS Circuit Representation
//
// An R1CS (Rank-1 Constraint System) consists of matrices A, B, C
// such that for a valid witness w:
//   (A · w) ⊙ (B · w) = (C · w)   (element-wise product)
//
// Each constraint is: <a_i, w> * <b_i, w> = <c_i, w>
// where a_i, b_i, c_i are sparse row vectors.
//
// For a neural network:
// - Each Fp multiplication becomes one R1CS constraint
// - Fp additions are free (absorbed into A, B, C coefficients)
// - Matrix multiply y = Wx + b generates O(M*N) constraints
//   (one per multiplication in each dot product)
//
// Witness layout:
// w = [1, public_inputs..., private_inputs..., intermediate_values...]
// w[0] = 1 (constant term)
// ============================================================

namespace zkml {

using bn254::Fr;

// Sparse matrix entry: (row, col, value)
struct SparseEntry {
    int row;
    int col;
    Fr value;
};

// R1CS constraint system
struct R1CS {
    int num_constraints;    // Number of R1CS constraints
    int num_variables;      // Total number of variables in witness
    int num_public_inputs;  // Number of public inputs (first entries after w[0]=1)
    int num_private_inputs; // Number of private inputs

    // Sparse matrices A, B, C
    std::vector<SparseEntry> A;
    std::vector<SparseEntry> B;
    std::vector<SparseEntry> C;

    R1CS() : num_constraints(0), num_variables(0),
             num_public_inputs(0), num_private_inputs(0) {}

    // Add a multiplication constraint:
    // (a_left_var * a_coeff) * (b_right_var * b_coeff) = (c_out_var * c_coeff)
    void add_mul_constraint(int a_var, Fr a_coeff,
                             int b_var, Fr b_coeff,
                             int c_var, Fr c_coeff) {
        int row = num_constraints;
        A.push_back({row, a_var, a_coeff});
        B.push_back({row, b_var, b_coeff});
        C.push_back({row, c_var, c_coeff});
        num_constraints++;
    }

    // Add a linear constraint: sum_i a_i * w[i] = sum_j c_j * w[j]
    // (B matrix has constant 1 at position w[0])
    void add_linear_constraint(const std::vector<std::pair<int, Fr>>& a_terms,
                                const std::vector<std::pair<int, Fr>>& c_terms) {
        int row = num_constraints;
        for (auto& [var, coeff] : a_terms) {
            A.push_back({row, var, coeff});
        }
        B.push_back({row, 0, Fr::one()}); // w[0] = 1
        for (auto& [var, coeff] : c_terms) {
            C.push_back({row, var, coeff});
        }
        num_constraints++;
    }

    // Verify that witness satisfies all constraints
    bool verify_witness(const std::vector<Fr>& witness) const {
        if ((int)witness.size() != num_variables) {
            printf("[R1CS] Witness size %d != expected %d\n",
                   (int)witness.size(), num_variables);
            return false;
        }

        // Compute A*w, B*w, C*w for each constraint
        std::vector<Fr> Aw(num_constraints, Fr::zero());
        std::vector<Fr> Bw(num_constraints, Fr::zero());
        std::vector<Fr> Cw(num_constraints, Fr::zero());

        for (const auto& e : A) {
            Aw[e.row] = Aw[e.row] + e.value * witness[e.col];
        }
        for (const auto& e : B) {
            Bw[e.row] = Bw[e.row] + e.value * witness[e.col];
        }
        for (const auto& e : C) {
            Cw[e.row] = Cw[e.row] + e.value * witness[e.col];
        }

        // Check: Aw ⊙ Bw = Cw
        for (int i = 0; i < num_constraints; i++) {
            Fr lhs = Aw[i] * Bw[i];
            if (!(lhs == Cw[i])) {
                printf("[R1CS] Constraint %d failed\n", i);
                return false;
            }
        }
        return true;
    }
};

// ============================================================
// Generate R1CS circuit from neural network structure
// ============================================================
struct CircuitBuilder {
    R1CS circuit;
    int next_var;

    CircuitBuilder() : next_var(1) {} // var 0 is reserved for constant 1

    // Allocate a new variable, returns its index
    int alloc_var() { return next_var++; }

    // Allocate n variables, returns index of first
    int alloc_vars(int n) {
        int first = next_var;
        next_var += n;
        return first;
    }

    // Build circuit for a linear layer: y = Wx + b
    // x_vars: indices of input variables
    // y_vars: indices of output variables (already allocated)
    std::vector<int> add_linear_layer_into(const std::vector<int>& x_vars,
                                           const std::vector<int>& y_vars,
                                           int out_size, int in_size) {
        // For each output y[i] = sum_j W[i][j] * x[j] + b[i]
        // We need intermediate variables for each product W[i][j] * x[j]
        for (int i = 0; i < out_size; i++) {
            std::vector<int> product_vars(in_size);
            for (int j = 0; j < in_size; j++) {
                product_vars[j] = alloc_var();
                // Constraint: w_ij * x_j = product_ij
                int w_var = alloc_var(); // Variable for weight (witness will fill)
                circuit.add_mul_constraint(
                    w_var, Fr::one(),
                    x_vars[j], Fr::one(),
                    product_vars[j], Fr::one()
                );
            }

            // Constraint: sum(products) + b = y[i]
            // This is a linear constraint (free in R1CS)
            std::vector<std::pair<int, Fr>> a_terms;
            for (int j = 0; j < in_size; j++) {
                a_terms.push_back({product_vars[j], Fr::one()});
            }
            int b_var = alloc_var(); // bias variable
            a_terms.push_back({b_var, Fr::one()});
            circuit.add_linear_constraint(a_terms, {{y_vars[i], Fr::one()}});
        }

        return y_vars;
    }

    // Build circuit for a linear layer: y = Wx + b
    // x_vars: indices of input variables
    // Returns indices of newly allocated output variables
    std::vector<int> add_linear_layer(const std::vector<int>& x_vars,
                                      int out_size, int in_size) {
        std::vector<int> y_vars(out_size);
        for (int i = 0; i < out_size; i++) {
            y_vars[i] = alloc_var();
        }
        return add_linear_layer_into(x_vars, y_vars, out_size, in_size);
    }

    // Build circuit for polynomial ReLU approximation
    // f(x) = c0 + c1*x + c2*x^2 + c3*x^3
    std::vector<int> add_relu_approx(const std::vector<int>& x_vars, int size,
                                     Fr c0, Fr c1, Fr c2, Fr c3, Fr scale) {
        std::vector<int> y_vars(size);
        Fr sc0 = c0 * scale;
        Fr sc1 = c1 * scale;
        Fr sc2 = c2 * scale;
        Fr sc3 = c3 * scale;
        for (int i = 0; i < size; i++) {
            y_vars[i] = alloc_var();

            // x^2
            int x2_var = alloc_var();
            circuit.add_mul_constraint(
                x_vars[i], Fr::one(),
                x_vars[i], Fr::one(),
                x2_var, Fr::one()
            );

            // x^3
            int x3_var = alloc_var();
            circuit.add_mul_constraint(
                x2_var, Fr::one(),
                x_vars[i], Fr::one(),
                x3_var, Fr::one()
            );

            // y = scale * (c0 + c1*x + c2*x^2 + c3*x^3)
            circuit.add_linear_constraint(
                {
                    {0, sc0},
                    {x_vars[i], sc1},
                    {x2_var, sc2},
                    {x3_var, sc3}
                },
                {{y_vars[i], Fr::one()}}
            );
        }
        return y_vars;
    }

    // Build a lightweight polynomial softmax approximation for a score row:
    // exp(x) ~= 1 + x + x^2 + x^3, y_i = exp_i / sum_j exp_j.
    std::vector<int> add_softmax_approx(const std::vector<int>& x_vars, int size) {
        std::vector<int> y_vars(size);
        std::vector<int> exp_vars(size);

        for (int i = 0; i < size; i++) {
            y_vars[i] = alloc_var();
            int x2_var = alloc_var();
            int x3_var = alloc_var();
            int exp_var = alloc_var();
            exp_vars[i] = exp_var;

            circuit.add_mul_constraint(
                x_vars[i], Fr::one(),
                x_vars[i], Fr::one(),
                x2_var, Fr::one()
            );

            circuit.add_mul_constraint(
                x2_var, Fr::one(),
                x_vars[i], Fr::one(),
                x3_var, Fr::one()
            );

            circuit.add_linear_constraint(
                {
                    {0, Fr::one()},
                    {x_vars[i], Fr::one()},
                    {x2_var, Fr::one()},
                    {x3_var, Fr::one()}
                },
                {{exp_var, Fr::one()}}
            );
        }

        int sum_var = alloc_var();
        std::vector<std::pair<int, Fr>> exp_terms;
        for (int i = 0; i < size; i++) {
            exp_terms.push_back({exp_vars[i], Fr::one()});
        }
        circuit.add_linear_constraint(exp_terms, {{sum_var, Fr::one()}});

        int sum_inv_var = alloc_var();
        circuit.add_mul_constraint(
            sum_var, Fr::one(),
            sum_inv_var, Fr::one(),
            0, Fr::one()
        );

        for (int i = 0; i < size; i++) {
            circuit.add_mul_constraint(
                exp_vars[i], Fr::one(),
                sum_inv_var, Fr::one(),
                y_vars[i], Fr::one()
            );
        }

        return y_vars;
    }

    // Build a self-attention block over flattened tokens.
    std::vector<int> add_self_attention_layer(const std::vector<int>& x_vars,
                                              int seq_len,
                                              int hidden_size,
                                              int num_heads = 1) {
        if (num_heads <= 0 || (hidden_size % num_heads) != 0) {
            return {};
        }
        const int head_dim = hidden_size / num_heads;
        auto alloc_matrix = [&](int rows, int cols) {
            std::vector<int> vars(rows * cols);
            for (int i = 0; i < rows * cols; i++) {
                vars[i] = alloc_var();
            }
            return vars;
        };

        auto alloc_vector = [&](int n) {
            std::vector<int> vars(n);
            for (int i = 0; i < n; i++) {
                vars[i] = alloc_var();
            }
            return vars;
        };

        std::vector<int> q_w = alloc_matrix(hidden_size, hidden_size);
        std::vector<int> q_b = alloc_vector(hidden_size);
        std::vector<int> k_w = alloc_matrix(hidden_size, hidden_size);
        std::vector<int> k_b = alloc_vector(hidden_size);
        std::vector<int> v_w = alloc_matrix(hidden_size, hidden_size);
        std::vector<int> v_b = alloc_vector(hidden_size);
        std::vector<int> o_w = alloc_matrix(hidden_size, hidden_size);
        std::vector<int> o_b = alloc_vector(hidden_size);

        auto add_shared_projection = [&](const std::vector<int>& weight_vars,
                                         const std::vector<int>& bias_vars,
                                         const std::vector<int>& input_flat) {
            std::vector<int> out_flat(seq_len * hidden_size);
            for (int token = 0; token < seq_len; token++) {
                for (int out = 0; out < hidden_size; out++) {
                    int y_var = alloc_var();
                    out_flat[token * hidden_size + out] = y_var;

                    std::vector<std::pair<int, Fr>> sum_terms;
                    for (int in = 0; in < hidden_size; in++) {
                        int product_var = alloc_var();
                        circuit.add_mul_constraint(
                            weight_vars[out * hidden_size + in], Fr::one(),
                            input_flat[token * hidden_size + in], Fr::one(),
                            product_var, Fr::one()
                        );
                        sum_terms.push_back({product_var, Fr::one()});
                    }
                    sum_terms.push_back({bias_vars[out], Fr::one()});
                    circuit.add_linear_constraint(sum_terms, {{y_var, Fr::one()}});
                }
            }
            return out_flat;
        };

        std::vector<int> q_vars = add_shared_projection(q_w, q_b, x_vars);
        std::vector<int> k_vars = add_shared_projection(k_w, k_b, x_vars);
        std::vector<int> v_vars = add_shared_projection(v_w, v_b, x_vars);

        std::vector<int> score_vars(seq_len * num_heads * seq_len);
        for (int token = 0; token < seq_len; token++) {
            for (int head = 0; head < num_heads; head++) {
                for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                    int score_var = alloc_var();
                    score_vars[(token * num_heads + head) * seq_len + key_idx] = score_var;

                    std::vector<std::pair<int, Fr>> score_terms;
                    for (int d = 0; d < head_dim; d++) {
                        int idx = head * head_dim + d;
                        int product_var = alloc_var();
                        circuit.add_mul_constraint(
                            q_vars[token * hidden_size + idx], Fr::one(),
                            k_vars[key_idx * hidden_size + idx], Fr::one(),
                            product_var, Fr::one()
                        );
                        score_terms.push_back({product_var, Fr::one()});
                    }
                    circuit.add_linear_constraint(score_terms, {{score_var, Fr::one()}});
                }
            }
        }

        std::vector<int> attn_probs(seq_len * num_heads * seq_len);
        for (int token = 0; token < seq_len; token++) {
            for (int head = 0; head < num_heads; head++) {
                std::vector<int> row_scores(seq_len);
                for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                    row_scores[key_idx] = score_vars[(token * num_heads + head) * seq_len + key_idx];
                }
                std::vector<int> row_probs = add_softmax_approx(row_scores, seq_len);
                for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                    attn_probs[(token * num_heads + head) * seq_len + key_idx] = row_probs[key_idx];
                }
            }
        }

        std::vector<int> context_vars(seq_len * hidden_size);
        for (int token = 0; token < seq_len; token++) {
            for (int head = 0; head < num_heads; head++) {
                for (int d = 0; d < head_dim; d++) {
                    int idx = head * head_dim + d;
                    int context_var = alloc_var();
                    context_vars[token * hidden_size + idx] = context_var;

                    std::vector<std::pair<int, Fr>> ctx_terms;
                    for (int key_idx = 0; key_idx < seq_len; key_idx++) {
                        int product_var = alloc_var();
                        circuit.add_mul_constraint(
                            attn_probs[(token * num_heads + head) * seq_len + key_idx], Fr::one(),
                            v_vars[key_idx * hidden_size + idx], Fr::one(),
                            product_var, Fr::one()
                        );
                        ctx_terms.push_back({product_var, Fr::one()});
                    }
                    circuit.add_linear_constraint(ctx_terms, {{context_var, Fr::one()}});
                }
            }
        }

        std::vector<int> out_vars(seq_len * hidden_size);
        for (int token = 0; token < seq_len; token++) {
            for (int out = 0; out < hidden_size; out++) {
                int y_var = alloc_var();
                out_vars[token * hidden_size + out] = y_var;

                std::vector<std::pair<int, Fr>> sum_terms;
                for (int in = 0; in < hidden_size; in++) {
                    int product_var = alloc_var();
                    circuit.add_mul_constraint(
                        o_w[out * hidden_size + in], Fr::one(),
                        context_vars[token * hidden_size + in], Fr::one(),
                        product_var, Fr::one()
                    );
                    sum_terms.push_back({product_var, Fr::one()});
                }
                sum_terms.push_back({o_b[out], Fr::one()});
                sum_terms.push_back({x_vars[token * hidden_size + out], Fr::one()});
                circuit.add_linear_constraint(sum_terms, {{y_var, Fr::one()}});
            }
        }

        return out_vars;
    }

    // Finalize the circuit
    void finalize(int num_public, int num_private) {
        circuit.num_variables = next_var;
        circuit.num_public_inputs = num_public;
        circuit.num_private_inputs = num_private;
    }
};

} // namespace zkml
