#include "gorby_sdpa.cuh"
#include "cuda_check.hpp"

#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>

using namespace gorby::utils;

#define CUTLASS_CHECK(status)                                                                    \
	{                                                                                              \
	cutlass::Status error = status;                                                              \
	if (error != cutlass::Status::kSuccess) {                                                    \
		std::cerr << "Got cutlass error: " << cutlassGetStatusString(error) << " at: " << __LINE__ \
				<< std::endl;                                                                    \
		exit(EXIT_FAILURE);                                                                        \
	}                                                                                            \
}

namespace gorby{
    namespace sdpa{
        // CUDA Binding - Simple CUTLASS SGEMM kernel
		// See https://github.com/NVIDIA/cutlass/issues/282 for example
        torch::Tensor cutlass_sgemm_nn(
			torch::Tensor A, 
			torch::Tensor B, 
			torch::Tensor C
		) {
			// Create output tensor D
			auto D_options = torch::TensorOptions()
				.dtype(C.dtype())
				.layout(C.layout())
				.device(C.device())
				.requires_grad(false);

			torch::Tensor D = torch::empty_like(C, D_options);

			// Get Tensor Accessors - assert type is float and A,B,C have 2 dimensions
			auto a = A.packed_accessor64<float, 2>();
			auto b = B.packed_accessor64<float, 2>();
			auto c = C.packed_accessor64<float, 2>();
			auto d = D.packed_accessor64<float, 2>();

			// Let's do the matrix multiplication via CUTLASS
			// Create GEMM instance
			using CutlassSGEMM_NNOperator = cutlass::gemm::device::Gemm<
				float, cutlass::layout::ColumnMajor,
				float, cutlass::layout::ColumnMajor,
				float, cutlass::layout::ColumnMajor,
				float,
				cutlass::arch::OpClassSimt,
				cutlass::arch::Sm86
				// This code section describes the tile size a thread block will compute
				// cutlass::gemm::GemmShape<128, 128, 1>,
				// This code section describes tile size a warp will compute
				// cutlass::gemm::GemmShape<64, 64, 1>,
				// This code section describes the size of MMA op
				// cutlass::gemm::GemmShape<32, 32, 1>
			>;

			CutlassSGEMM_NNOperator cutlass_sgemm_nn_operator_instance;

			// Get Problem Size
			int M = (int) A.size(0);
			int N = (int) A.size(1);
			int K = (int) C.size(1);

			CutlassSGEMM_NNOperator::Arguments args(
				{M, N, K},
				{(float*)A.data_ptr(), cutlass::layout::ColumnMajor(a.stride(0))},
				{(float*)B.data_ptr(), cutlass::layout::ColumnMajor(b.stride(0))},
				{(float*)C.data_ptr(), cutlass::layout::ColumnMajor(c.stride(0))},
				{(float*)D.data_ptr(), cutlass::layout::ColumnMajor(d.stride(0))},
				{1.0f, 1.0f}
			);
			
			// Invoke the CUTLASS GEMM template
			cutlass::Status status = cutlass_sgemm_nn_operator_instance(args);
			CUTLASS_CHECK(status);

			// Return!
			return D;
		}


        // Definitions
        torch::Tensor gorby_sdpa_forward(
        	torch::Tensor A, torch::Tensor B, torch::Tensor C
        ) {
			CHECK_CUDA(A);
			CHECK_CUDA(B);
			CHECK_CUDA(C);
			CHECK_CONTIGUOUS(A);
			CHECK_CONTIGUOUS(B);
			CHECK_CONTIGUOUS(C);
			CHECK_SAME_TYPE(A, B);
			CHECK_SAME_TYPE(B, C);
			TORCH_CHECK(A.size(1) == B.size(0) && C.size(0) == A.size(0) && C.size(1) == B.size(1));

			torch::Tensor D = cutlass_sgemm_nn(A, B, C);

			return D;
		}
    }
}