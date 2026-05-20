#ifndef SPARSE_ENGINES_CUDA_CSRC_CUDA_COMMON_CUH
#define SPARSE_ENGINES_CUDA_CSRC_CUDA_COMMON_CUH

#include <c10/util/BFloat16.h>
#include <c10/util/Half.h>

#define MAX_THREADS_PER_BLOCK 1024

namespace sparse_engines_cuda {

// Upcast load helper: fp16 / bf16 / fp32 → fp32 for register-resident accum.
// c10::Half and c10::BFloat16 have explicit conversion operators to float;
// for fp32 the cast is a no-op.
__device__ __forceinline__ float to_float_src(float x)            { return x; }
__device__ __forceinline__ float to_float_src(c10::Half x)        { return static_cast<float>(x); }
__device__ __forceinline__ float to_float_src(c10::BFloat16 x)    { return static_cast<float>(x); }

} // namespace sparse_engines_cuda

#endif // SPARSE_ENGINES_CUDA_CSRC_CUDA_COMMON_CUH