#include <torch/extension.h>

// Definitions
torch::Tensor gorby_vector_add_1d_inplace( torch::Tensor x, torch::Tensor y );

// PyBind!
PYBIND11_MODULE( TORCH_EXTENSION_NAME, m ) {
  m.def("gorby_vector_add_1d_inplace", &gorby_vector_add_1d_inplace, "CUDA 1D Vector Add, Inplace");
}