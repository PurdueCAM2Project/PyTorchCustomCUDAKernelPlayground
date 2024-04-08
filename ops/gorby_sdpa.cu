#include "gorby_sdpa.cuh"

// CUDA
#include <cuda.h>
#include <cuda_runtime.h>

// CUTLASS
#include <cutlass/numeric_types.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/util/host_tensor.h>

// STDLIB
#include <vector>

namespace gorby{
    namespace sdpa{
        // Definitions
        torch::Tensor gorby_fused_tome_sdpa_forward(torch::Tensor x, int32_t r){
            return x;
        }
    }
}