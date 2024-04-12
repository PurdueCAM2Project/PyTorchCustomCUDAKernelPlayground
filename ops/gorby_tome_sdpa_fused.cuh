#pragma once
#include <torch/extension.h>

namespace gorby{
    namespace tome{
        // Definitions
        torch::Tensor gorby_fused_tome_sdpa_forward(
            torch::Tensor x, 
            int32_t r
        );

        // PyBind!
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
            m.def("gorby_fused_tome_sdpa_forward", &gorby_fused_tome_sdpa_forward, "(Dense) Scaled Dot Product Attention (SDPA)");
        }
    }
}