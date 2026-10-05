// Host-only shim so the field / R1CS headers can be compiled and unit-tested
// without the CUDA toolkit (tests/host). Never used by the real CUDA build.
#pragma once
#include <cstddef>
#define __host__
#define __device__
#define __global__
#define __forceinline__ inline
#define __constant__
typedef int cudaError_t;
static const cudaError_t cudaSuccess = 0;
static const cudaError_t cudaErrorMemoryAllocation = 2;
inline const char* cudaGetErrorString(cudaError_t) { return "host-shim"; }
inline cudaError_t cudaMemGetInfo(size_t* f, size_t* t) { *f = (size_t)1 << 32; *t = (size_t)1 << 32; return 0; }
inline cudaError_t cudaMalloc(void**, size_t) { return 2; }
template <typename T> inline cudaError_t cudaMalloc(T**, size_t) { return 2; }
typedef void* cudaEvent_t;
inline cudaError_t cudaEventCreate(cudaEvent_t* e) { *e = nullptr; return 0; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return 0; }
inline cudaError_t cudaEventRecord(cudaEvent_t, int = 0) { return 0; }
inline cudaError_t cudaEventSynchronize(cudaEvent_t) { return 0; }
inline cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) { *ms = 0.f; return 0; }
struct cudaDeviceProp { char name[256]; size_t totalGlobalMem; int major; int minor; int multiProcessorCount; };
inline cudaError_t cudaGetDeviceProperties(cudaDeviceProp* p, int) { p->name[0] = 0; p->totalGlobalMem = 0; p->major = p->minor = 0; p->multiProcessorCount = 0; return 0; }
inline cudaError_t cudaGetDevice(int* d) { *d = 0; return 0; }
inline cudaError_t cudaDeviceSynchronize() { return 0; }
struct c3d_dim3_shim { unsigned x = 0, y = 0, z = 0; };
static const c3d_dim3_shim blockIdx{}, blockDim{}, threadIdx{}, gridDim{};
