#include "gorby_nvtx.hpp"

#include <string>
#include <nvtx3/nvToolsExt.h>

// Scope with namespaces
namespace gorby{
    namespace nvtx{
        // Initialize NVTX manually - otherwise we will incur overhead at the first instance of an NVTX call
        void nvtx_init(void){
            nvtxInitialize(0);
        }

        // Instantaneous NVTX marker
        void nvtx_mark(std::string marker_name){
            nvtxMark(marker_name.c_str());
        }

        // Handle Ranges!
        // Add range
        void nvtx_range_push(std::string range_name){
            nvtxRangePushA(range_name.c_str());
        }

        // Remove / pop range
        void nvtx_range_pop(void){
            nvtxRangePop();
        }
    }
}