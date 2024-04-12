#pragma once
#include <torch/extension.h>

namespace gorby{
    namespace sdpa{
        // Definitions
        torch::Tensor gorby_sdpa_forward(
            torch::Tensor A, torch::Tensor B, torch::Tensor C
        );

        // PyBind!
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
            m.def("sdpa_forward", &gorby_sdpa_forward, "(Dense) Scaled Dot Product Attention (SDPA)");
        }
    }
}