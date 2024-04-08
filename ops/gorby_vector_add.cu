#include "gorby_vector_add.cuh"

#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

namespace gorby{
    namespace vector_add{
        // CUDA Kernel Checking Helpers
        // Put in anonymous namespace to not pollute anything - not sure if this would be a problem, but it seems proper
        namespace
        {
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

            void __inline__ CHECK_INPUT(torch::Tensor x){
                CHECK_CUDA(x);
                CHECK_CONTIGUOUS(x);
            }
        }

        // CUDA Binding
        template <typename scalar_t>
        __global__ void gorby_vector_add_1d_inplace_cuda_kernel(
        // We could handle raw pointers to data directly - but we would have to handle byte level data (stride, offset, data type?) - this is gross
        // This is an accessor helper class that handles different types, and throws exception for wrong type / shape
        // Handling this in CUDA directly would be a mess, so this helper is useful!
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> x,
        torch::PackedTensorAccessor32<scalar_t, 1, torch::RestrictPtrTraits> y,
        unsigned int length)
        {
            const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;

            // Check index - make sure its not greater than or equal to length
            if (index < length)
            {
                x[index] = x[index] + y[index];
            }
        }

        // CPU Binding
        template <typename scalar_t>
        void gorby_vector_add_1d_inplace_cpu_kernel(
        torch::TensorAccessor<scalar_t, 1> x,
        torch::TensorAccessor<scalar_t, 1> y,
        unsigned int length)
        {
            for (unsigned int i = 0; i < length; i++)
            {
                x[i] = x[i] + y[i];
            }
        }

        // Helper function for CUDA kernel - choose # threads per block based on size of tensor
        unsigned int __inline__ get_threads_per_block_from_tensor_dim(torch::Tensor x, int dim)
        {
            unsigned int threads_per_block;
            const auto tensor_size_along_dim = x.size(dim);

            // NOTE: Minimal working size for threads is 32 (warp) see CUDA programming guide for this info
            // No sense in using something smaller than this
            if (tensor_size_along_dim <= 32){
                threads_per_block = 32;
            }
            else if (tensor_size_along_dim <= 64){
                threads_per_block = 64;
            }
            else if (tensor_size_along_dim <= 128){
                threads_per_block = 128;
            }
            else if (tensor_size_along_dim <= 256){
                threads_per_block = 256;
            }
            else if (tensor_size_along_dim <= 512){
                threads_per_block = 512;
            }
            // Max threads per block is 1024 for most NVIDIA GPU architectures - see CUDA programming guide
            // Lets use 1024 for the maximum size if our length is > 1024 - we will just dispatch more thread blocks then
            else{
                threads_per_block = 1024;
            }

            return threads_per_block;
        }

        // This function does the heavy lifting
        // Depending on the device type, we invoke a CUDA kernel or a simple C++ implementation for CPU tensors
        // Note, we can beautify this and make it simpler by registering our functions with the PyTorch dispatcher (TODO)
        torch::Tensor _gorby_vector_add_1d_inplace(
        torch::Tensor x,
        torch::Tensor y){
            // Let's compute this so we can dispatch small thread blocks for smaller tensor sizes (more efficient, less wasted operations)
            const unsigned int threads_per_block = get_threads_per_block_from_tensor_dim(x, 0);

            // Reasoning: For lengths < 1024, we launch just 1 block (+1 handles this, the 2nd term will be 0 via integer division)
            // For sizes > 1024, we can launch multiple blocks (the 2nd term will be >= 1 )
            const unsigned int block_x = 1 + ((unsigned int)x.size(0) - 1) / threads_per_block;

            // This is the proper algorithm - for x size <= 2048 we have a single block
            // For larger x sizes we will simply have more blocks!
            const dim3 blocks(block_x, 1, 1);

            // Dispatch based on dtype!
            if (x.device().type() == torch::kCUDA){
                // Lambda function, but the call to the kernel is done via CUDA kernel launch syntax <<<...>>>
                // NOTE: This assumes floating point data. See ATen/Dispatch.h for other options!
                AT_DISPATCH_FLOATING_TYPES(x.type(), "gorby_vector_add_1d_inplace_cuda", ([&]
                                                                                        { gorby_vector_add_1d_inplace_cuda_kernel<scalar_t><<<blocks, threads_per_block>>>(
                                                                                                x.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                                y.packed_accessor32<scalar_t, 1, torch::RestrictPtrTraits>(),
                                                                                                (unsigned int)x.size(0)); }));
            }
            else{
                // NOTE: This assumes floating point data. See ATen/Dispatch.h for other options!
                AT_DISPATCH_FLOATING_TYPES(x.type(), "gorby_vector_add_1d_inplace_cpu", ([&]
                                                                                        { gorby_vector_add_1d_inplace_cpu_kernel<scalar_t>(
                                                                                            x.accessor<scalar_t, 1>(),
                                                                                            y.accessor<scalar_t, 1>(),
                                                                                            (unsigned int)x.size(0)); }));
            }

            // Return x - though we don't necessarily need to since this is an in-place operation
            return x;
        }

        // This is the function we will bind to Python
        // Here we call the helper function with a similar name to do the heavy lifting, but first do some input validation
        torch::Tensor gorby_vector_add_1d_inplace(
        torch::Tensor x,
        torch::Tensor y){
            // Input Checking! What fun
            CHECK_SAME_TYPE(x, y);
            CHECK_SAME_DEVICE(x, y);
            CHECK_LENGTH_ALONG_DIM_NONZERO(x, y, 0);
            CHECK_CONTIGUOUS(x);
            CHECK_CONTIGUOUS(y);

            return _gorby_vector_add_1d_inplace(x, y);
        }
    }
}