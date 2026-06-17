#include <cmath>
#include <cstring>

#include "cuda_runtime.h"
// #include "cuda_profiler_api.h"

#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __DEVICE_FUNCTIONS_HPP__
#define __CUDACC__
#include "crt/device_functions.h"  // some __cuda* functions
#undef __CUDACC__

#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__
#define __CUDA_INTERNAL_COMPILATION__
#define __COMMON_FUNCTIONS_H__
#include "crt/host_runtime.h"  // the rest __cuda* functions
