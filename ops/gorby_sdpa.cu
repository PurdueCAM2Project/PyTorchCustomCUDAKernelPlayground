#include "gorby_sdpa.cuh"
#include "cuda_check.hpp"

// Inspired by this partially
// https://github.com/imoneoi/cutlass_grouped_gemm/blob/main/csrc/grouped_gemm.cu

// CUTLASS
#include <cutlass/gemm/device/gemm.h>

// STDLIB
#include <iostream>
#include <sstream>
#include <vector>

// Namespace using stuff
using namespace gorby::utils;

namespace gorby{
    namespace sdpa{
        // Private functions used to implement the kernel(s)
        namespace{
            // Recall format of GEMM operation is: (alpha * A @ B) + (beta * C)
            // A is M x K
            // B is K x N
            // C is M x N
            // S prefix is single-precision
            // NN means A and B are column major
            cudaError_t CutlassSgemmNN(
                int M,
                int N,
                int K,
                float alpha,
                float const *A,
                int lda,
                float const *B,
                int ldb,
                float beta,
                float *C,
                int ldc
            ){
                cutlass::gemm::device::Gemm<
                float,
                cutlass::layout::ColumnMajor,
                float,
                cutlass::layout::ColumnMajor,
                float,
                cutlass::layout::ColumnMajor
                > gemm_op;

                //
                // Launch the GEMM operation on the device
                //

                cutlass::Status status = gemm_op({
                {M, N, K},                          // GemmCoord problem_size,
                {A, lda},                           // TensorRef<float, layout::ColumnMajor> ref_A,
                {B, ldb},                           // TensorRef<float, layout::ColumnMajor> ref_B,
                {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_C,
                {C, ldc},                           // TensorRef<float, layout::ColumnMajor> ref_D,
                {alpha, beta}                       // EpilogueOutputOp::Params epilogue_op_params
                });

                return status;
                /*
                cutlass::Status status = op({
                    {M, N, K},
                    {A, lda},           // TensorRef to A device tensor
                    {B, ldb},           // TensorRef to B device tensor
                    {C, ldc},           // TensorRef to C device tensor
                    {C, ldc},           // Tensor-ref for destination matrix D (may be different memory than source C matrix)
                    {alpha, beta}       // epilogue operation arguments
                });

                if (status != cutlass::Status::kSuccess) {
                    return cudaErrorUnknown;
                }
                // Return success, if no errors were encountered.
                return cudaSuccess;
                */
            }

            // Wrapper for _CUTLASS_SgemmNN_kernel_launch that decomposes tensor options into raw pointer data
            cudaError_t _CUTLASS_SgemmNN_kernel_launch_from_pytorch(
                const torch::Tensor& A,
                const torch::Tensor& B,
                const torch::Tensor& C,
                float alpha,
                float beta
            ){
                // First, get sizes and check
                auto A_shape = A.sizes();
                auto B_shape = B.sizes();
                auto C_shape = C.sizes();

                utils::CHECK_CUDA(A);
                utils::CHECK_CUDA(B);
                utils::CHECK_CUDA(C);

                utils::CHECK_CONTIGUOUS(A);
                utils::CHECK_CONTIGUOUS(B);
                utils::CHECK_CONTIGUOUS(C);

                utils::CHECK_SAME_TYPE(A, B);
                utils::CHECK_SAME_TYPE(B, C);
                
                return cudaSuccess;
            }
        }


        // Definitions
        torch::Tensor gorby_sdpa_forward(torch::Tensor A, torch::Tensor B, torch::Tensor C){
            cudaError_t status = _CUTLASS_SgemmNN_kernel_launch_from_pytorch(A, B, C, 1.0, 1.0);
            TORCH_CHECK(status == cudaSuccess);
            return A;
        }
    }
}