#pragma once
#include <torch/extension.h>

namespace gorby{
    namespace sdpa{
        // Definitions
        torch::Tensor gorby_sdpa_forward(
			torch::Tensor A, torch::Tensor B, torch::Tensor C
        );

        torch::Tensor gorby_softmax_forward(
			torch::Tensor A
        );

        // PyBind!
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
            m.def("sdpa_forward", &gorby_sdpa_forward, "(Dense) Scaled Dot Product Attention (SDPA)");
            m.def("softmax_forward", &gorby_softmax_forward, "Softmax Forward");
        }
    }
}