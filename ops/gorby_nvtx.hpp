#pragma once
#include <torch/extension.h>

namespace gorby{
    namespace nvtx{
        // Definitions
        // Initialize NVTX manually - otherwise we will incur overhead at the first instance of an NVTX call
        void nvtx_init(void);
        void nvtx_mark(std::string marker_name);
        void nvtx_range_push(std::string range_name);
        void nvtx_range_pop(void);

        // PyBind!
        PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
            m.def("nvtx_init", &nvtx_init, "Initialize NVTX marker");
            m.def("nvtx_mark", &nvtx_mark, "Add NVTX marker");
            m.def("nvtx_range_push", &nvtx_range_push, "Start / Push NVTX Named Range");
            m.def("nvtx_range_pop", &nvtx_range_pop, "End / Pop NVTX Named Range");
        }
    }
}