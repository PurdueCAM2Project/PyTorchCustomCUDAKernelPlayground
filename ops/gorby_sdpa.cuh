#include <torch/extension.h>

namespace gorby{
    namespace sdpa{
        // Definitions
        torch::Tensor gorby_sdpa_forward(
            torch::Tensor x, 
            int32_t r
        );

        // PyBind!
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
            m.def("sdpa_forward", &gorby_sdpa_forward, "Scaled Dot Product Attention (SDPA)");
        }
    }
}