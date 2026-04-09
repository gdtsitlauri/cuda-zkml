cmake_minimum_required(VERSION 3.18)
project(cuda-zkml LANGUAGES CXX CUDA)

option(ZKML_FAST_DEBUG_LOOP "Speed up iterative prover/debug builds" ON)
option(ZKML_ULTRA_FAST_COMPILE "Minimize CUDA compile time for rapid iteration (slower runtime)" ON)
set(ZKML_NVCC_THREADS "1" CACHE STRING "NVCC internal thread count (0 means auto)")
set(ZKML_NVCC_SPLIT_COMPILE_THREADS "1" CACHE STRING "NVCC split-compile thread count (0 means auto)")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 75)

option(ZKML_FAST_ITERATION "Use faster CUDA compile settings for debug iteration" ON)

if(MSVC)
    # CUDA 13 + CCCL requires standard-conforming MSVC preprocessor.
    add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>")
endif()

if(NOT ZKML_NVCC_THREADS MATCHES "^[0-9]+$")
    message(FATAL_ERROR "ZKML_NVCC_THREADS must be a non-negative integer")
endif()

if(NOT ZKML_NVCC_SPLIT_COMPILE_THREADS MATCHES "^[0-9]+$")
    message(FATAL_ERROR "ZKML_NVCC_SPLIT_COMPILE_THREADS must be a non-negative integer")
endif()

# CUDA flags for GTX 1650 (sm_75, Turing)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda --threads ${ZKML_NVCC_THREADS} --split-compile ${ZKML_NVCC_SPLIT_COMPILE_THREADS}")
if(MSVC)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=/Zc:preprocessor")
endif()
if(ZKML_FAST_ITERATION)
    # Faster ptxas turnaround for active development/debug loops.
    if(ZKML_ULTRA_FAST_COMPILE)
        set(CMAKE_CUDA_FLAGS_RELEASE "-O0")
    else()
        set(CMAKE_CUDA_FLAGS_RELEASE "-O1 -lineinfo")
    endif()
else()
    set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math")
endif()
set(CMAKE_CUDA_FLAGS_DEBUG "-G -g -lineinfo")

# Find CUDA
find_package(CUDAToolkit REQUIRED)

# Include directories
include_directories(${CMAKE_SOURCE_DIR}/src)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

# ============================================================
# Core library: field arithmetic
# ============================================================
add_library(zkml_field STATIC
    src/field/fp.cu
)
set_target_properties(zkml_field PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(zkml_field CUDA::cudart)

# ============================================================
# Core library: curve operations
# ============================================================
add_library(zkml_curve STATIC
    src/curve/g1.cu
    src/curve/g2.cu
)
set_target_properties(zkml_curve PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(zkml_curve zkml_field CUDA::cudart)

# ============================================================
# Core library: NTT
# ============================================================
add_library(zkml_ntt STATIC
    src/ntt/ntt.cu
)
set_target_properties(zkml_ntt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(zkml_ntt zkml_field CUDA::cudart)

# ============================================================
# Core library: MSM
# ============================================================
add_library(zkml_msm STATIC
    src/msm/msm.cu
)
set_target_properties(zkml_msm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(zkml_msm zkml_curve zkml_field CUDA::cudart)

# ============================================================
# Core library: NN inference
# ============================================================
add_library(zkml_nn STATIC
    src/nn/layers.cu
    src/nn/inference.cu
)
set_target_properties(zkml_nn PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(zkml_nn zkml_field CUDA::cudart)

# ============================================================
# Core library: Prover
# ============================================================
add_library(zkml_prover STATIC
    src/prover/groth16.cu
)
set_target_properties(zkml_prover PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(zkml_prover zkml_msm zkml_ntt zkml_curve zkml_field zkml_nn CUDA::cudart)

# ============================================================
# Core library: Verifier
# ============================================================
add_library(zkml_verifier STATIC
    src/verifier/verifier.cu
)
set_target_properties(zkml_verifier PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(zkml_verifier zkml_curve zkml_field CUDA::cudart)

# ============================================================
# CLI tools
# ============================================================
add_executable(zkml-prove src/cli/prove.cu)
set_target_properties(zkml-prove PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(zkml-prove zkml_prover zkml_verifier zkml_nn zkml_msm zkml_ntt zkml_curve zkml_field CUDA::cudart)

add_executable(zkml-verify src/cli/verify.cu)
set_target_properties(zkml-verify PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(zkml-verify zkml_prover zkml_verifier zkml_curve zkml_field CUDA::cudart)

if(ZKML_FAST_DEBUG_LOOP)
    message(STATUS "ZKML_FAST_DEBUG_LOOP=ON (using faster CUDA compile flags for prover targets)")
    if(ZKML_ULTRA_FAST_COMPILE)
        message(STATUS "ZKML_ULTRA_FAST_COMPILE=ON (O0 compile for prover/CLI targets)")
    endif()

    if(ZKML_ULTRA_FAST_COMPILE)
        target_compile_options(zkml_prover PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O0>
        )
        target_compile_options(zkml-prove PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O0>
        )
        target_compile_options(zkml-verify PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O0>
        )
    else()
        target_compile_options(zkml_prover PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O1>
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        )
        target_compile_options(zkml-prove PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O1>
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        )
        target_compile_options(zkml-verify PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O1>
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        )
    endif()
endif()

# ============================================================
# Tests
# ============================================================
enable_testing()

add_executable(test_field tests/test_field.cu)
set_target_properties(test_field PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_field zkml_field CUDA::cudart)
add_test(NAME FieldTest COMMAND test_field)

add_executable(test_curve tests/test_curve.cu)
set_target_properties(test_curve PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_curve zkml_curve zkml_field CUDA::cudart)
add_test(NAME CurveTest COMMAND test_curve)

add_executable(test_ntt tests/test_ntt.cu)
set_target_properties(test_ntt PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_ntt zkml_ntt zkml_field CUDA::cudart)
add_test(NAME NTTTest COMMAND test_ntt)

add_executable(test_msm tests/test_msm.cu)
set_target_properties(test_msm PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_msm zkml_msm zkml_curve zkml_field CUDA::cudart)
add_test(NAME MSMTest COMMAND test_msm)

add_executable(test_groth16 tests/test_groth16.cu)
set_target_properties(test_groth16 PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(test_groth16 PRIVATE zkml_prover zkml_verifier zkml_msm zkml_ntt zkml_curve zkml_field zkml_nn CUDA::cudart)
add_test(NAME Groth16Test COMMAND test_groth16)

if(ZKML_FAST_DEBUG_LOOP)
    if(ZKML_ULTRA_FAST_COMPILE)
        # Μην αλλάζεις το CUDA_SEPARABLE_COMPILATION για το test_groth16, κράτα το ON για σωστό linking
        target_compile_options(test_groth16 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O0>
        )
    else()
        target_compile_options(test_groth16 PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-O1>
            $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
        )
    endif()
endif()

# ============================================================
# Python bindings (optional, requires pybind11)
# ============================================================
find_package(pybind11 QUIET)
if(pybind11_FOUND)
    pybind11_add_module(zkml_binding python/binding.cpp)
    target_link_libraries(zkml_binding PRIVATE
        zkml_prover zkml_verifier zkml_nn zkml_msm zkml_ntt zkml_curve zkml_field
        CUDA::cudart
    )
    set_target_properties(zkml_binding PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/zkml
    )
endif()

message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "NVCC threading: --threads ${ZKML_NVCC_THREADS}, --split-compile ${ZKML_NVCC_SPLIT_COMPILE_THREADS}")
