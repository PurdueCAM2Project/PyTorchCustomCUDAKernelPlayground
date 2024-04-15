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

        void __inline__ CHECK_NONZERO_DIM(const torch::Tensor &x, int dim){
            TORCH_CHECK(x.size(dim) > 0, x, " must have nonzero size along dim", dim);
        }

        void __inline__ CHECK_MATRIX_SHAPE(const torch::Tensor &x){
            TORCH_CHECK(x.sizes().size() == 2, x, " must have exactly two dimensions");
        }

        void __inline__ CHECK_VECTOR_SHAPE(const torch::Tensor &x){
            TORCH_CHECK(x.sizes().size() == 1, x, " must have exactly one dimension");
        }
    }
}