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
            m.def("vector_add_1d_inplace", &gorby_vector_add_1d_inplace, "CUDA 1D Vector Add, Inplace");
        }
    }
}