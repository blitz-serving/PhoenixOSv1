#include <cmath>
#include <cstring>

#include "cuda_runtime.h"
#include "cuda_profiler_api.h"

// https://github.com/llvm/llvm-project/blob/main/clang/lib/Headers/__clang_cuda_runtime_wrapper.h

extern "C" unsigned __cudaPushCallConfiguration(
    dim3 gridDim,
    dim3 blockDim,
    size_t sharedMem = 0,
    struct CUstream_st *stream = 0
);

#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_INTERNAL_COMPILATION__
#define __COMMON_FUNCTIONS_H__
#include "crt/host_runtime.h"  // the rest __cuda* functions
