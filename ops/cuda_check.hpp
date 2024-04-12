#pragma once
#include <torch/extension.h>

namespace gorby{
    namespace utils{
        void __inline__ CHECK_CUDA(const torch::Tensor &x){
            TORCH_CHECK(x.device().is_cuda(), x, " must be a CUDA tensor");
        }

        void __inline__ CHECK_CONTIGUOUS(const torch::Tensor &x){
            TORCH_CHECK(x.is_contiguous(), x, " must be contiguous");
        }

        void __inline__ CHECK_LENGTH_ALONG_DIM_NONZERO(const torch::Tensor &x, const torch::Tensor &y, int dim){
            TORCH_CHECK(x.size(dim) == y.size(dim), x, " and ", y, " must have same size along dim ", dim);
        }

        void __inline__ CHECK_SAME_TYPE(const torch::Tensor &x, const torch::Tensor &y){
            TORCH_CHECK(x.dtype() == y.dtype(), x, " must have same type as ", y);
        }

        void __inline__ CHECK_SAME_DEVICE(const torch::Tensor &x, const torch::Tensor &y){
            TORCH_CHECK(x.device() == y.device(), x, " must have same device as ", y);
        }
    }
}