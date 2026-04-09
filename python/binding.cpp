/*
 * pybind11 bindings for CUDA-zkML
 *
 * Exposes core proving and verification functions to Python.
 * Requires pybind11: pip install pybind11
 *
 * Build: automatically handled by CMakeLists.txt if pybind11 is found.
 */

#ifdef PYBIND11_AVAILABLE

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
namespace fs = std::filesystem;

static std::string quote_arg(const std::string& arg) {
    std::string escaped;
    escaped.reserve(arg.size() + 2);
    for (char c : arg) {
        if (c == '"') {
            escaped.push_back('\\');
        }
        escaped.push_back(c);
    }
    return std::string("\"") + escaped + "\"";
}

static void run_command_or_throw(const std::string& cmd) {
    int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::ostringstream oss;
        oss << "Command failed (exit=" << rc << "): " << cmd;
        throw std::runtime_error(oss.str());
    }
}

static std::vector<uint8_t> read_binary_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open file: " + path);
    }
    in.seekg(0, std::ios::end);
    std::streamoff size = in.tellg();
    in.seekg(0, std::ios::beg);
    if (size < 0) {
        throw std::runtime_error("Failed to determine file size: " + path);
    }
    std::vector<uint8_t> data((size_t)size);
    if (size > 0) {
        in.read(reinterpret_cast<char*>(data.data()), size);
    }
    return data;
}

static std::vector<uint64_t> read_public_inputs_limb0(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Failed to open public inputs file: " + path);
    }

    int32_t n = 0;
    in.read(reinterpret_cast<char*>(&n), sizeof(int32_t));
    if (!in || n < 0) {
        throw std::runtime_error("Invalid public inputs header: " + path);
    }

    std::vector<uint64_t> out;
    out.reserve((size_t)n);
    for (int32_t i = 0; i < n; i++) {
        uint64_t limbs[4] = {0, 0, 0, 0};
        in.read(reinterpret_cast<char*>(limbs), sizeof(limbs));
        if (!in) {
            throw std::runtime_error("Truncated public inputs file: " + path);
        }
        out.push_back(limbs[0]);
    }
    return out;
}

static std::string join_path(const std::string& base, const std::string& leaf) {
    return (fs::path(base) / fs::path(leaf)).string();
}

// Wrapper classes for Python
struct PyProof {
    std::vector<uint8_t> data;
    std::vector<uint64_t> public_inputs;
    bool valid;

    PyProof() : valid(false) {}

    py::bytes to_bytes() const {
        return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
    }

    static PyProof from_bytes(const py::bytes& b) {
        PyProof proof;
        std::string s = b;
        proof.data.assign(s.begin(), s.end());
        proof.valid = true;
        return proof;
    }
};

struct PyVerificationKey {
    std::vector<uint8_t> data;
    bool valid;

    PyVerificationKey() : valid(false) {}

    py::bytes to_bytes() const {
        return py::bytes(reinterpret_cast<const char*>(data.data()), data.size());
    }

    static PyVerificationKey from_file(const std::string& path) {
        PyVerificationKey vk;
        vk.data = read_binary_file(path);
        vk.valid = !vk.data.empty();
        return vk;
    }
};

static PyProof load_proof_bundle(const std::string& proof_path,
                                 const std::string& public_inputs_path) {
    PyProof proof;
    proof.data = read_binary_file(proof_path);
    proof.public_inputs = read_public_inputs_limb0(public_inputs_path);
    proof.valid = !proof.data.empty();
    return proof;
}

// Run demo pipeline via CLI and return generated proof artifacts.
PyProof demo_prove(const std::string& prove_bin = "zkml-prove",
                   const std::string& work_dir = ".") {
    fs::create_directories(work_dir);

#if defined(_WIN32)
    std::string cmd = std::string("cd /d ") + quote_arg(work_dir) +
                      " && " + quote_arg(prove_bin) + " --demo";
#else
    std::string cmd = std::string("cd ") + quote_arg(work_dir) +
                      " && " + quote_arg(prove_bin) + " --demo";
#endif

    run_command_or_throw(cmd);

    return load_proof_bundle(
        join_path(work_dir, "proof.bin"),
        join_path(work_dir, "public_inputs.bin"));
}

// Run proving from model/input files via CLI and return generated proof bytes.
PyProof prove_from_files(const std::string& model_path,
                         const std::string& input_path,
                         const std::string& prove_bin = "zkml-prove",
                         const std::string& work_dir = ".") {
    fs::create_directories(work_dir);

    std::string proof_path = join_path(work_dir, "proof.bin");
    std::string vk_path = join_path(work_dir, "vk.bin");
    std::string pi_path = join_path(work_dir, "public_inputs.bin");

    std::string cmd =
        quote_arg(prove_bin) +
        " --model " + quote_arg(model_path) +
        " --input " + quote_arg(input_path) +
        " --output " + quote_arg(proof_path) +
        " --vk " + quote_arg(vk_path) +
        " --public-inputs " + quote_arg(pi_path);

    run_command_or_throw(cmd);

    return load_proof_bundle(proof_path, pi_path);
}

// Verify proof files via CLI; returns true on success.
bool verify_files(const std::string& vk_path,
                  const std::string& proof_path,
                  const std::string& public_inputs_path,
                  const std::string& verify_bin = "zkml-verify") {
    std::string cmd =
        quote_arg(verify_bin) +
        " --vk " + quote_arg(vk_path) +
        " --proof " + quote_arg(proof_path) +
        " --public-inputs " + quote_arg(public_inputs_path);

    return std::system(cmd.c_str()) == 0;
}

PYBIND11_MODULE(zkml_binding, m) {
    m.doc() = "CUDA-zkML: GPU-Accelerated ZK Proofs for Neural Networks";

    py::class_<PyProof>(m, "Proof")
        .def(py::init<>())
        .def_readonly("valid", &PyProof::valid)
        .def_readonly("public_inputs", &PyProof::public_inputs)
        .def("to_bytes", &PyProof::to_bytes)
        .def_static("from_bytes", &PyProof::from_bytes)
        .def("__repr__", [](const PyProof& p) {
            return "<Proof size=" + std::to_string(p.data.size()) +
                   " valid=" + (p.valid ? "True" : "False") + ">";
        });

    py::class_<PyVerificationKey>(m, "VerificationKey")
        .def(py::init<>())
                .def_readonly("valid", &PyVerificationKey::valid)
                .def("to_bytes", &PyVerificationKey::to_bytes)
                .def_static("from_file", &PyVerificationKey::from_file,
                                        py::arg("path"));

    m.def("demo_prove", &demo_prove,
                    py::arg("prove_bin") = "zkml-prove",
                    py::arg("work_dir") = ".",
                    "Run demo proving pipeline and return proof bytes.");

        m.def("prove_from_files", &prove_from_files,
                    py::arg("model_path"),
                    py::arg("input_path"),
                    py::arg("prove_bin") = "zkml-prove",
                    py::arg("work_dir") = ".",
                    "Run proving from model/input files via CLI and return proof bytes.");

        m.def("verify_files", &verify_files,
                    py::arg("vk_path"),
                    py::arg("proof_path"),
                    py::arg("public_inputs_path"),
                    py::arg("verify_bin") = "zkml-verify",
                    "Verify proof files via CLI.");

    m.attr("__version__") = "0.1.0";
}

#else
// Compilation stub when pybind11 is not available
// This file is only compiled if pybind11 is found by CMake
#endif
