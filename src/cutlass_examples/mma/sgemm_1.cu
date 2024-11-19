/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include "gemm.h"

#include <cute/util/debug.hpp>
#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <thrust/device_vector.h>

namespace v1 {

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class AThreadLayout,
          class TB, class BStride, class BSmemLayout, class BThreadLayout,
          class TC, class CStride, class CSmemLayout, class CThreadLayout,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(cute::size(CThreadLayout{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, AThreadLayout tA,
            TB const* B, BStride dB, BSmemLayout sB_layout, BThreadLayout tB,
            TC      * C, CStride dC, CSmemLayout          , CThreadLayout tC,
            Alpha alpha, Beta beta)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(cute::rank(shape_MNK) == cute::Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(cute::rank(cta_tiler) == cute::Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  static_assert(cute::is_static<AThreadLayout>::value);
  static_assert(cute::is_static<BThreadLayout>::value);
  static_assert(cute::is_static<CThreadLayout>::value);

  // Layouts may be different but they must have the same size
  CUTE_STATIC_ASSERT_V(cute::size(tA) == cute::size(tB));                          // NumThreads
  CUTE_STATIC_ASSERT_V(cute::size(tC) == cute::size(tA));                          // NumThreads

  CUTE_STATIC_ASSERT_V(cute::size<0>(cta_tiler) % cute::size<0>(tA) == cute::Int<0>{});  // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(cute::size<2>(cta_tiler) % cute::size<1>(tA) == cute::Int<0>{});  // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(cute::size<1>(cta_tiler) % cute::size<0>(tB) == cute::Int<0>{});  // BLK_N / THR_N
  CUTE_STATIC_ASSERT_V(cute::size<2>(cta_tiler) % cute::size<1>(tB) == cute::Int<0>{});  // BLK_K / THR_K
  CUTE_STATIC_ASSERT_V(cute::size<0>(cta_tiler) % cute::size<0>(tC) == cute::Int<0>{});  // BLK_M / THR_M
  CUTE_STATIC_ASSERT_V(cute::size<1>(cta_tiler) % cute::size<1>(tC) == cute::Int<0>{});  // BLK_N / THR_N

  static_assert(cute::is_static<ASmemLayout>::value);
  static_assert(cute::is_static<BSmemLayout>::value);
  static_assert(cute::is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(cute::size<0>(ASmemLayout{}) == cute::size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(cute::size<0>(CSmemLayout{}) == cute::size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(cute::size<0>(BSmemLayout{}) == cute::size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(cute::size<1>(CSmemLayout{}) == cute::size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(cute::size<1>(ASmemLayout{}) == cute::size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(cute::size<1>(BSmemLayout{}) == cute::size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(cute::select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(cute::select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(cute::select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  cute::Tensor mA = cute::make_tensor(cute::make_gmem_ptr(A), cute::select<0,2>(shape_MNK), dA); // (M,K)
  cute::Tensor mB = cute::make_tensor(cute::make_gmem_ptr(B), cute::select<1,2>(shape_MNK), dB); // (N,K)
  cute::Tensor mC = cute::make_tensor(cute::make_gmem_ptr(C), cute::select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = cute::make_coord(blockIdx.x, blockIdx.y, cute::_);              // (m,n,k)
  cute::Tensor gA = cute::local_tile(mA, cta_tiler, cta_coord, cute::Step<cute::_1, cute::X, cute::_1>{});  // (BLK_M, BLK_K, k)
  cute::Tensor gB = cute::local_tile(mB, cta_tiler, cta_coord, cute::Step<cute::X, cute::_1, cute::_1>{});  // (BLK_N, BLK_K, k)
  cute::Tensor gC = cute::local_tile(mC, cta_tiler, cta_coord, cute::Step<cute::_1, cute::_1, cute::X>{});  // (BLK_M, BLK_N)

  // Shared memory buffers
  __shared__ TA smemA[cute::cosize_v<ASmemLayout>];
  __shared__ TB smemB[cute::cosize_v<BSmemLayout>];
  cute::Tensor sA = cute::make_tensor(cute::make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  cute::Tensor sB = cute::make_tensor(cute::make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of simple raked partitioning of ThreadLayouts tA|tB over data A|B tiles

  cute::Tensor tAgA = cute::local_partition(gA, tA, threadIdx.x);                  // (THR_M,THR_K,k)
  cute::Tensor tAsA = cute::local_partition(sA, tA, threadIdx.x);                  // (THR_M,THR_K)

  cute::Tensor tBgB = cute::local_partition(gB, tB, threadIdx.x);                  // (THR_N,THR_K,k)
  cute::Tensor tBsB = cute::local_partition(sB, tB, threadIdx.x);                  // (THR_N,THR_K)

  CUTE_STATIC_ASSERT_V(cute::size<0>(tAgA) == cute::size<0>(tAsA));                // THR_M
  CUTE_STATIC_ASSERT_V(cute::size<1>(tAgA) == cute::size<1>(tAsA));                // THR_K
  CUTE_STATIC_ASSERT_V(cute::size<0>(tBgB) == cute::size<0>(tBsB));                // THR_N
  CUTE_STATIC_ASSERT_V(cute::size<1>(tBgB) == cute::size<1>(tBsB));                // THR_K

  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via projections of a ThreadLayout tC

  // Partition sA (M,K) by the rows of tC
  cute::Tensor tCsA = cute::local_partition(sA, tC, threadIdx.x, cute::Step<cute::_1, cute::X>{});   // (THR_M,BLK_K)

  // Partition sB (N,K) by the cols of tC
  cute::Tensor tCsB = cute::local_partition(sB, tC, threadIdx.x, cute::Step<cute::X, cute::_1>{});   // (THR_N,BLK_K)

  // Partition gC (M,N) by the tile of tC
  cute::Tensor tCgC = cute::local_partition(gC, tC, threadIdx.x, cute::Step<cute::_1, cute::_1>{});   // (THR_M,THR_N)

  // Allocate the accumulators -- same shape/layout as the partitioned data
  cute::Tensor tCrC = make_tensor_like(tCgC);                                // (THR_M,THR_N)

  CUTE_STATIC_ASSERT_V(cute::size<0>(tCrC) == cute::size<0>(tCgC));                // THR_M
  CUTE_STATIC_ASSERT_V(cute::size<0>(tCrC) == cute::size<0>(tCsA));                // THR_M
  CUTE_STATIC_ASSERT_V(cute::size<1>(tCrC) == cute::size<1>(tCgC));                // THR_N
  CUTE_STATIC_ASSERT_V(cute::size<1>(tCrC) == cute::size<0>(tCsB));                // THR_N
  CUTE_STATIC_ASSERT_V(cute::size<1>(tCsA) == cute::size<1>(tCsB));                // BLK_K

  // Clear the accumulators
  clear(tCrC);

#if 1

  // TUTORIAL: Example of a simple mainloop that read tiles of data into shared memory,
  //           and then computes on those tiles.
  //   cute::copy(.) operates on the global and shared memory via the tA|tB partitioning
  //   gemm(.) operates on the shared and register memory via the tC partitioning

  auto K_TILE_MAX = cute::size<2>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
#if 1
    if (cute::thread0()) {
      printf("\n\n");
      cute::print("  mA : "); cute::print(  mA); cute::print("\n");
      cute::print("  gA : "); cute::print(  gA); cute::print("\n");
      cute::print("  sA : "); cute::print(  sA); cute::print("\n");
      cute::print("tAgA : "); cute::print(tAgA); cute::print("\n");
      cute::print("tAsA : "); cute::print(tAsA); cute::print("\n");

      cute::print("  mB : "); cute::print(  mB); cute::print("\n");
      cute::print("  gB : "); cute::print(  gB); cute::print("\n");
      cute::print("  sB : "); cute::print(  sB); cute::print("\n");
      cute::print("tBgB : "); cute::print(tBgB); cute::print("\n");
      cute::print("tBsB : "); cute::print(tBsB); cute::print("\n");

      cute::print("  mC : "); cute::print(  mC); cute::print("\n");
      cute::print("  gC : "); cute::print(  gC); cute::print("\n");
      cute::print("tCsA : "); cute::print(tCsA); cute::print("\n");
      cute::print("tCsB : "); cute::print(tCsB); cute::print("\n");
      cute::print("tCgC : "); cute::print(tCgC); cute::print("\n");
      cute::print("tCrC : "); cute::print(tCrC); cute::print("\n");
      printf("\n\n");
    }
#endif

    // Clear tAsA and tBsB
    clear(tAsA);
    clear(tBsB);

    // Copy gmem to smem with tA|tB thread-partitioned tensors
    cute::copy(tAgA(cute::_, cute::_, k_tile), tAsA);      // A   (THR_M,THR_K) -> (THR_M,THR_K)
    cute::copy(tBgB(cute::_, cute::_, k_tile), tBsB);      // B   (THR_N,THR_K) -> (THR_N,THR_K)

    // TUTORIAL: The above call to cute::copy(tAgA(cute::_,cute::_,k_tile), tAsA) is equivalent to
    //   cute::Tensor tAgAk = tAgA(cute::_,cute::_,k_tile);
    //   CUTE_UNROLL
    //   for (int i = 0; i < cute::size(tAsA); ++i) {
    //     tAsA(i) = tAgAk(i);
    //   }

    cute::cp_async_fence();        // Label the end of (potential) cp.async instructions
    cute::cp_async_wait<0>();      // Sync on all (potential) cp.async instructions
    __syncthreads();         // Wait for all threads to write to smem

    // Compute gemm on tC thread-partitioned smem
    cute::gemm(tCsA, tCsB, tCrC);            // (THR_M,THR_N) += (THR_M,BLK_K) * (THR_N,BLK_K)

    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < cute::size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < cute::size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < cute::size<1>(tCrC); ++n) {
    //         tCrC(m,n) += tCsA(m,k) * tCsB(n,k);
    //       }
    //     }
    //   }

    __syncthreads();         // Wait for all threads to read from smem
  }

#endif

  //
  // Epilogue
  //

  cute::axpby(alpha, tCrC, beta, tCgC);

  // TUTORIAL: The above call to axpby(alpha, tCrC, beta, tCgC) is equivalent to
  //   CUTE_UNROLL
  //   for (int i = 0; i < cute::size(tCsA); ++i) {
  //     tCgC(i) = alpha * tCrC(i) + beta * tCgC(i);
  //   }
}

// Setup params for an NT GEMM
// Use m-major smem sA, n-major smem sB, and mn-major threads tA|tB
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_nt(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = cute::make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = cute::make_stride(cute::Int<1>{}, ldA);                      // (dM, dK)
  auto dB = cute::make_stride(cute::Int<1>{}, ldB);                      // (dN, dK)
  auto dC = cute::make_stride(cute::Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = cute::Int<8>{};
  auto bN = cute::Int<8>{};
  auto bK = cute::Int<4>{};
  auto cta_tiler = cute::make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = cute::make_layout(cute::make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
  auto sB = cute::make_layout(cute::make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
  auto sC = cute::make_layout(cute::make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA = cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<4>{}));   // (m,k) -> thr_idx
  auto tB = cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<4>{}));   // (n,k) -> thr_idx
  auto tC = cute::make_layout(cute::make_shape(cute::Int<4>{}, cute::Int<4>{}));   // (m,n) -> thr_idx

  dim3 threadsPerBlock(cute::size(tC));
  dim3 dimGrid(cute::size(cute::ceil_div(M, bM)),
               cute::size(cute::ceil_div(N, bN)));
  gemm_device<<<dimGrid, threadsPerBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}

// Setup params for a TN GEMM
// Use padded m-major smem sA, padded n-major smem sB, and k-major threads tA|tB
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
gemm_tn(int m, int n, int k,
        Alpha alpha,
        TA const* A, int ldA,
        TB const* B, int ldB,
        Beta beta,
        TC      * C, int ldC,
        cudaStream_t stream = 0)
{
  using namespace cute;

  // Define shapes (dynamic)
  auto M = int(m);
  auto N = int(n);
  auto K = int(k);
  auto prob_shape = cute::make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = cute::make_stride(ldA, cute::Int<1>{});                      // (dM, dK)
  auto dB = cute::make_stride(ldB, cute::Int<1>{});                      // (dN, dK)
  auto dC = cute::make_stride(cute::Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = cute::Int<128>{};
  auto bN = cute::Int<128>{};
  auto bK = cute::Int<  8>{};
  auto cta_tiler = cute::make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = cute::make_layout(cute::make_shape(bM,bK), LayoutRight{});   // (m,k) -> smem_idx; k-major
  auto sB = cute::make_layout(cute::make_shape(bN,bK), LayoutRight{});   // (n,k) -> smem_idx; k-major
  auto sC = cute::make_layout(cute::make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)
  auto tA = cute::make_layout(cute::make_shape(cute::Int<32>{}, cute::Int< 8>{}), LayoutRight{});  // (m,k) -> thr_idx; k-major
  auto tB = cute::make_layout(cute::make_shape(cute::Int<32>{}, cute::Int< 8>{}), LayoutRight{});  // (n,k) -> thr_idx; k-major
  auto tC = cute::make_layout(cute::make_shape(cute::Int<16>{}, cute::Int<16>{}));                 // (m,n) -> thr_idx; m-major

  dim3 threadsPerBlock(cute::size(tC));
  dim3 dimGrid(cute::size(cute::ceil_div(M, bM)),
               cute::size(cute::ceil_div(N, bN)));
  gemm_device<<<dimGrid, threadsPerBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, tA,
       B, dB, sB, tB,
       C, dC, sC, tC,
       alpha, beta);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
test_mma_v1(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     thrust::host_vector<TA> const& h_A, int ldA,
     thrust::host_vector<TB> const& h_B, int ldB,
     Beta beta,
     thrust::host_vector<TC>      & h_C, int ldC)
{
  cudaStream_t stream = 0;

  // Copy input matrices
  thrust::device_vector<TA> d_A = h_A;
  thrust::device_vector<TB> d_B = h_B;
  thrust::device_vector<TC> d_C = h_C;

  if (transA == 'N' && transB == 'T') {
    gemm_nt(m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC, stream);
  } else
  if (transA == 'T' && transB == 'N') {
    gemm_tn(m, n, k, alpha, d_A.data().get(), ldA, d_B.data().get(), ldB, beta, d_C.data().get(), ldC, stream);
  }

  // Copy output matrix
  h_C = d_C;

  CUTE_CHECK_LAST();
}

template void test_mma_v1<float, float, float, float, float>(char transA, char transB, int m, int n, int k,
                                                      float alpha,
                                                      thrust::host_vector<float> const& h_A, int ldA,
                                                      thrust::host_vector<float> const& h_B, int ldB,
                                                      float beta,
                                                      thrust::host_vector<float>      & h_C, int ldC);

} // namespace v1
