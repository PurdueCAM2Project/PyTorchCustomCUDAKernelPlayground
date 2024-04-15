#include "gorby_sdpa.cuh"
#include "cuda_check.hpp"

#include <algorithm>
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/gemm_coord.h>

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
				cutlass::arch::Sm80
				// This code section describes the tile size a thread block will compute
				// cutlass::gemm::GemmShape<128, 256, 64>
				// This code section describes tile size a warp will compute
				//cutlass::gemm::GemmShape<64, 64, 32>
				// This code section describes the size of MMA op
				// cutlass::gemm::GemmShape<32, 32, 1>
			>;

			// // A matrix configuration
			// using         ElementA    = cutlass::half_t;                                // Element type for A matrix operand
			// using         LayoutA     = cutlass::layout::RowMajor;                      // Layout type for A matrix operand
			// constexpr int AlignmentA  = 128 / cutlass::sizeof_bits<ElementA>::value;    // Memory access granularity/alignment of A matrix in units of elements (up to 16 bytes)

			// // B matrix configuration
			// using         ElementB    = cutlass::half_t;                                // Element type for B matrix operand
			// using         LayoutB     = cutlass::layout::RowMajor;                      // Layout type for B matrix operand
			// constexpr int AlignmentB  = 128 / cutlass::sizeof_bits<ElementB>::value;    // Memory access granularity/alignment of B matrix in units of elements (up to 16 bytes)

			// // C/D matrix configuration
			// using         ElementC    = cutlass::half_t;                                // Element type for C and D matrix operands
			// using         LayoutC     = cutlass::layout::RowMajor;                      // Layout type for C and D matrix operands
			// constexpr int AlignmentC  = 128 / cutlass::sizeof_bits<ElementC>::value;    // Memory access granularity/alignment of C/D matrices in units of elements (up to 16 bytes)

			// // Multiply-accumulate blocking/pipelining details
			// using ElementAccumulator  = cutlass::half_t;                          // Element type for internal accumulation
			// using ArchTag             = cutlass::arch::Sm80;                      // Tag indicating the minimum SM that supports the intended feature
			// using OperatorClass       = cutlass::arch::OpClassTensorOp;           // Operator class tag
			// using ThreadblockShape    = cutlass::gemm::GemmShape<128, 128, 32>;   // Threadblock-level tile size (concept: GemmShape)
			// using WarpShape           = cutlass::gemm::GemmShape<64, 64, 32>;     // Warp-level tile size (concept: GemmShape)
			// using InstructionShape    = cutlass::gemm::GemmShape<16, 8, 16>;      // Instruction-level tile size (concept: GemmShape)
			// constexpr int NumStages   = 4;                                        // Number of global->shared pipeline stages used in the GEMM mainloop

			// // StreamK device GEMM implementation type
			// using CutlassSGEMM_NNOperator = cutlass::gemm::device::GemmUniversal<
			// 	float, cutlass::layout::ColumnMajor,
			// 	float, cutlass::layout::ColumnMajor,
			// 	float, cutlass::layout::ColumnMajor,
			// 	float,
			// 	cutlass::arch::OpClassTensorOp,
			// 	cutlass::arch::Sm80,
			// 	ThreadblockShape,
			// 	WarpShape,
			// 	InstructionShape,
			// 	EpilogueOp,
			// 	cutlass::gemm::threadblock::ThreadblockSwizzleStreamK, // <-- Only difference
			// 	NumStages,
			// 	AlignmentA,
			// 	AlignmentB
			// >;

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

		// Softmax Sum and reduction
		__device__ __inline__ void _exp_max_subtraction_fp32_matrix_n(
        	torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> a,
        	torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> b,
			unsigned int M,
			unsigned int N,
			float maximum_a
		) {
			// Get x, y indidces
            const unsigned int x_index = threadIdx.x + (blockIdx.x * blockDim.x);
			const unsigned int y_index = threadIdx.y + (blockIdx.y * blockDim.y);
			if(x_index < M && y_index < N) {
				b[x_index][y_index] = __expf(a[x_index][y_index] - maximum_a);
			}
		}

		// Device Binding for single precision (FP32) softmax with 2D Matrix (M, N) as input - reduction performed along N (columns) dimension
		// MxN matrix "a" input
		// MxN matrix "b" output
		__global__ void softmax_fp32_matrix_n_kernel(
        	torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> a,
        	torch::PackedTensorAccessor64<float, 2, torch::RestrictPtrTraits> b,
			unsigned int M,
			unsigned int N,
			float maximum_a
		) {
			_exp_max_subtraction_fp32_matrix_n(a, b, M, N, maximum_a);
		}

		// Host Binding for Softmax
		torch::Tensor softmax_matrix_n(
			torch::Tensor A
		) {
			// Create output tensor B
			auto B_options = torch::TensorOptions()
				.dtype(A.dtype())
				.layout(A.layout())
				.device(A.device())
				.requires_grad(false);

			// Initialize
			torch::Tensor B = torch::empty_like(A, B_options);

			// Get maximum of A
			auto maximum_a = torch::max(A);

			// Select kernel dimensions
			unsigned int M = A.size(0);
			unsigned int N = A.size(1);

			// Threads per block = 32 * 32 * 1 = 1024
            const dim3 threadBlockDimension(32, 32, 1);
			const unsigned int gridDimX = (M+32-1) / 32;
			const unsigned int gridDimY = (N+32-1) / 32;
            const dim3 gridDimension(gridDimX, gridDimY, 1);

			// Launch kernel
			// AT_DISPATCH_FLOATING_TYPES(
			// 	A.type(), 
			// 	"softmax_fp32_matrix_n_kernel", ([&]{ 
			// 		softmax_fp32_matrix_n_kernel<scalar_t><<<gridDimension, threadBlockDimension>>>(
			// 			A.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(), 
			// 			B.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
			// 			M,
			// 			N,
			// 			maximum_a
			// 		);
			// 	}
			// ));

			// Launch Kernel - not AT_DISPATCH wrapper
			softmax_fp32_matrix_n_kernel<<<gridDimension, threadBlockDimension>>>(
				A.packed_accessor64<float, 2, torch::RestrictPtrTraits>(), 
				B.packed_accessor64<float, 2, torch::RestrictPtrTraits>(),
				M,
				N,
				maximum_a.item<float>()
			);

			// Return!
			return B;
		}

        // Definitions
        torch::Tensor gorby_sdpa_forward(
        	torch::Tensor A, torch::Tensor B, torch::Tensor C
        ) {
			CHECK_MATRIX_SHAPE(A);
			CHECK_MATRIX_SHAPE(B);
			CHECK_MATRIX_SHAPE(C);
			CHECK_CUDA(A);
			CHECK_CUDA(B);
			CHECK_CUDA(C);
			CHECK_CONTIGUOUS(A);
			CHECK_CONTIGUOUS(B);
			CHECK_CONTIGUOUS(C);
			CHECK_SAME_TYPE(A, B);
			CHECK_SAME_TYPE(B, C);
			TORCH_CHECK(A.size(1) == B.size(0) && C.size(0) == A.size(0) && C.size(1) == B.size(1));
			CHECK_NONZERO_DIM(A, 0);
			CHECK_NONZERO_DIM(A, 1);
			CHECK_NONZERO_DIM(B, 0);
			CHECK_NONZERO_DIM(B, 1);
			CHECK_NONZERO_DIM(C, 0);
			CHECK_NONZERO_DIM(C, 1);

			torch::Tensor D = cutlass_sgemm_nn(A, B, C);

			return D;
		}

        torch::Tensor gorby_softmax_forward(
			torch::Tensor A
        ) {
			CHECK_MATRIX_SHAPE(A);
			CHECK_CUDA(A);
			CHECK_CONTIGUOUS(A);
			CHECK_NONZERO_DIM(A, 0);
			CHECK_NONZERO_DIM(A, 1);

			torch::Tensor B = softmax_matrix_n(A);
			
			return B;
		}
    }
}