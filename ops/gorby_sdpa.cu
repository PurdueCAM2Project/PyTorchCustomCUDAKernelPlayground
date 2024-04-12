// CUTLASS
#include "gorby_sdpa.cuh"
#include "cutlass/cutlass.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/gemm/device/gemm_array.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/detail/helper_macros.hpp"

cudaError_t cutlass_array_sgemm(
  int m,
  int n,
  int k,
  float alpha,
  float const * const *A,
  int lda,
  float const * const *B,
  int ldb,
  float * const *C,
  int ldc,
  float beta,
  int batch_count) {
  #if (CUTLASS_CXX17_OR_LATER==1)
  #pragma message ( "gorby_sdpa.cu: C++17 Detected!" )
  using Gemm = cutlass::gemm::device::GemmArray<
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor,
    float, cutlass::layout::ColumnMajor
  >;

  Gemm gemm_op;

  cutlass::Status status = gemm_op({
    {m, n, k},
    A, lda,
    B, ldb,
    C, ldc,
    C, ldc,
    {alpha, beta},
    batch_count
  });

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  #else
  #define AUX(x) #x
  #define STRINGIFY(x) AUX(x)  
  #pragma message ( "gorby_sdpa.cu: C++17 Not Detected" )
  #pragma message ( STRINGIFY(__cplusplus) )
  #pragma message ( STRINGIFY(_MSVC_LANG) )
  #endif

  return cudaSuccess;
}