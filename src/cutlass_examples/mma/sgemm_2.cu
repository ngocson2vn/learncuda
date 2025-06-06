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

#include <cstdio>

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/util/debug.hpp>

#include "cutlass/util/print_error.hpp"
#include "cutlass/util/GPU_Clock.hpp"

#include <thrust/device_vector.h>


FILE* get_file_ptr() {
  FILE *fptr;
  const char* tex_file = "layout.tex";
  fptr = fopen(tex_file, "w");
  if(fptr == NULL) {
    printf("Failed to open %s\n", tex_file);
    exit(1);
  }

  return fptr;
}

// Generic 2D Layout to LaTeX printer
template <class LayoutA, class TikzColorFn = cute::TikzColor_BWx8>
CUTE_HOST_DEVICE
void
fprint_latex(FILE* os, LayoutA const& layout_a,   // (m,n) -> idx
                       TikzColorFn color = {})    // lambda(idx) -> tikz color string
{
  using namespace cute;
  CUTE_STATIC_ASSERT_V(rank(layout_a) <= Int<2>{});
  auto layout = append<2>(layout_a, Layout<_1,_0>{});

  // Header
  fprintf(os, "\\documentclass[convert]{standalone}\n"
         "\\usepackage{tikz}\n\n"
         "\\begin{document}\n"
         "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  // Layout
  for (int i = 0; i < size<0>(layout); ++i) {
    for (int j = 0; j < size<1>(layout); ++j) {
      int idx = layout(i,j);
      fprintf(os, "\\node[fill=%s] at (%d,%d) {%d};\n",
              color(idx), i, j, idx);
    }
  }
  // Grid
  fprintf(os, "\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid (%d,%d);\n\n",
         int(size<0>(layout)), int(size<1>(layout)));
  // Labels
  for (int i =  0, j = -1; i < size<0>(layout); ++i) {
    fprintf(os, "\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, i);
  }
  for (int i = -1, j =  0; j < size<1>(layout); ++j) {
    fprintf(os, "\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", i, j, j);
  }

  // Footer
  fprintf(os, "\\end{tikzpicture}\n"
         "\\end{document}\n");
}

// MNK MMA Layout to LaTeX TikZ
template <class LayoutC, class ThrIDC,
          class LayoutA, class ThrIDA,
          class LayoutB, class ThrIDB,
          class TikzColorFn = cute::TikzColor_TV>
CUTE_HOST_DEVICE
void
fprint_latex_mma(FILE* file_ptr, LayoutC const& C, ThrIDC const& TC,  // (m,n) -> (tid,vid)  and  tid -> thr_idx
                 LayoutA const& A, ThrIDA const& TA,  // (m,k) -> (tid,vid)  and  tid -> thr_idx
                 LayoutB const& B, ThrIDB const& TB,  // (n,k) -> (tid,vid)  and  tid -> thr_idx
                 TikzColorFn color = {})              // lambda(thr_idx,val_idx) -> tikz color string
{
  using namespace cute;
  CUTE_STATIC_ASSERT_V(rank(C) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(A) == Int<2>{});
  CUTE_STATIC_ASSERT_V(rank(B) == Int<2>{});

  assert(size<0>(A) == size<0>(C));
  assert(size<0>(B) == size<1>(C));
  assert(size<1>(A) == size<1>(B));

  // Header
  fprintf(file_ptr,
          "\\documentclass[convert]{standalone}\n"
          "\\usepackage{tikz}\n\n"
          "\\begin{document}\n"
          "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every node/.style={minimum size=1cm, outer sep=0pt}]\n\n");

  // C starting at 0,0
  for (int m = 0; m < size<0>(C); ++m) {
    for (int n = 0; n < size<1>(C); ++n) {
      int thrid   = C(m,n) % size(TC);
      int val_idx = C(m,n) / size(TC);
      int thr_idx = TC(thrid);

      fprintf(file_ptr, "\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color(thr_idx, val_idx),
              m, n,
              thr_idx, val_idx);
    }
  }
  // Grid
  fprintf(file_ptr, "\\draw[color=black,thick,shift={(-0.5,-0.5)}] (%d,%d) grid (%d,%d);\n\n",
          0, 0, int(size<0>(C)), int(size<1>(C)));

  // A starting at 0,-size<1>(A)-1
  for (int m = 0; m < size<0>(A); ++m) {
    for (int k = 0; k < size<1>(A); ++k) {
      int thrid   = A(m,k) % size(TA);
      int val_idx = A(m,k) / size(TA);
      int thr_idx = TA(thrid);

      fprintf(file_ptr, "\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
              color(thr_idx, val_idx),
              m, k-1-size<1>(A),
              thr_idx, val_idx);
    }
  }
  // Grid
  fprintf(file_ptr, "\\draw[color=black,thick,shift={(-0.5,-0.5)}] (%d,%d) grid (%d,%d);\n\n",
          0, int(-size<1>(A)-1), int(size<0>(A)), -1);
  // A labels
  for (int m =  0, k = -1; m < size<0>(A); ++m) {
    fprintf(file_ptr, "\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), m);
  }
  for (int m = -1, k =  0; k < size<1>(A); ++k) {
    fprintf(file_ptr, "\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", m, k-1-size<1>(A), k);
  }

  // B starting at -size<1>(B)-1,0
  for (int n = 0; n < size<0>(B); ++n) {
    for (int k = 0; k < size<1>(B); ++k) {
      int thrid   = B(n,k) % size(TB);
      int val_idx = B(n,k) / size(TB);
      int thr_idx = TB(thrid);

      fprintf(file_ptr, "\\node[fill=%s] at (%d,%d) {\\shortstack{T%d \\\\ V%d}};\n",
             color(thr_idx, val_idx),
             k-1-size<1>(B), n,
             thr_idx, val_idx);
    }
  }
  // Grid
  fprintf(file_ptr, "\\draw[color=black,thick,shift={(-0.5,-0.5)}] (%d,%d) grid (%d,%d);\n\n",
         int(-size<1>(B)-1), 0, -1, int(size<0>(B)));
  // B labels
  for (int n =  0, k = -1; n < size<0>(B); ++n) {
    fprintf(file_ptr, "\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, n);
  }
  for (int n = -1, k =  0; k < size<1>(B); ++k) {
    fprintf(file_ptr, "\\node at (%d,%d) {\\Large{\\texttt{%d}}};\n", k-1-size<1>(B), n, k);
  }

  // Footer
  fprintf(file_ptr, 
          "\\end{tikzpicture}\n"
          "\\end{document}\n");
}

// TiledMMA to LaTeX TikZ
template <class... Args, class TikzColorFn = cute::TikzColor_TV>
CUTE_HOST_DEVICE
void
fprint_latex(FILE* os, cute::TiledMMA<Args...> const& mma,
             TikzColorFn color = {})             // lambda(thr_idx,val_idx) -> tikz color string
{
  using namespace cute;
  auto layout_and_thrid_C = mma.get_layoutC_MN();
  auto layoutC_MN = get<0>(layout_and_thrid_C);
  auto thrID_C    = get<1>(layout_and_thrid_C);

  auto layout_and_thrid_A = mma.get_layoutA_MK();
  auto layoutA_MK = get<0>(layout_and_thrid_A);
  auto thrID_A    = get<1>(layout_and_thrid_A);

  auto layout_and_thrid_B = mma.get_layoutB_NK();
  auto layoutB_NK = get<0>(layout_and_thrid_B);
  auto thrID_B    = get<1>(layout_and_thrid_B);

  fprint_latex_mma(os, layoutC_MN, thrID_C,
                   layoutA_MK, thrID_A,
                   layoutB_NK, thrID_B);
}

namespace v2 {

template <class ProblemShape, class CtaTiler,
          class TA, class AStride, class ASmemLayout, class TiledCopyA,
          class TB, class BStride, class BSmemLayout, class TiledCopyB,
          class TC, class CStride, class CSmemLayout, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, AStride dA, ASmemLayout sA_layout, TiledCopyA copy_a,
            TB const* B, BStride dB, BSmemLayout sB_layout, TiledCopyB copy_b,
            TC      * C, CStride dC, CSmemLayout          , TiledMma mma,
            Alpha alpha, Beta beta)
{
  using namespace cute;

  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  CUTE_STATIC_ASSERT_V(size(copy_a) == size(mma));                     // NumThreads
  CUTE_STATIC_ASSERT_V(size(copy_b) == size(mma));                     // NumThreads

  static_assert(is_static<ASmemLayout>::value);
  static_assert(is_static<BSmemLayout>::value);
  static_assert(is_static<CSmemLayout>::value);

  CUTE_STATIC_ASSERT_V(size<0>(ASmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(CSmemLayout{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(BSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(CSmemLayout{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(ASmemLayout{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(BSmemLayout{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,2>(shape_MNK), dA));         // dA strides for shape MK
  CUTE_STATIC_ASSERT_V(congruent(select<1,2>(shape_MNK), dB));         // dB strides for shape NK
  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  Tensor mA = make_tensor(make_gmem_ptr(A), select<0,2>(shape_MNK), dA); // (M,K)
  Tensor mB = make_tensor(make_gmem_ptr(B), select<1,2>(shape_MNK), dB); // (N,K)
  Tensor mC = make_tensor(make_gmem_ptr(C), select<0,1>(shape_MNK), dC); // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k)
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory buffers
  __shared__ TA smemA[cosize_v<ASmemLayout>];
  __shared__ TB smemB[cosize_v<BSmemLayout>];
  Tensor sA = make_tensor(make_smem_ptr(smemA), sA_layout);            // (BLK_M,BLK_K)
  Tensor sB = make_tensor(make_smem_ptr(smemB), sB_layout);            // (BLK_N,BLK_K)

  //
  // Partition the copying of A and B tiles across the threads
  //

  // TUTORIAL: Example of partitioning via a TiledCopy

  ThrCopy thr_copy_a = copy_a.get_slice(threadIdx.x);
  Tensor tAgA = thr_copy_a.partition_S(gA);                            // (CPY,CPY_M,CPY_K,k)
  Tensor tAsA = thr_copy_a.partition_D(sA);                            // (CPY,CPY_M,CPY_K)

  // Allocate registers same shape/layout as partitioned data
  Tensor tArA = make_fragment_like(tAsA);                              // (CPY,CPY_M,CPY_K)

  ThrCopy thr_copy_b = copy_b.get_slice(threadIdx.x);
  Tensor tBgB = thr_copy_b.partition_S(gB);                            // (CPY,CPY_N,CPY_K,k)
  Tensor tBsB = thr_copy_b.partition_D(sB);                            // (CPY,CPY_N,CPY_K)

  // Allocate registers same shape/layout as partitioned data
  Tensor tBrB = make_fragment_like(tBsB);                              // (CPY,CPY_N,CPY_K)

  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tAsA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<1>(tAgA) == size<1>(tArA));                // CPY_M
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tAsA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tAgA) == size<2>(tArA));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBsB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<1>(tBgB) == size<1>(tBrB));                // CPY_N
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBsB));                // CPY_K
  CUTE_STATIC_ASSERT_V(size<2>(tBgB) == size<2>(tBrB));                // CPY_K

  // Copy gmem to rmem for k_tile=0
  copy(copy_a, tAgA(_,_,_,0), tArA);
  copy(copy_b, tBgB(_,_,_,0), tBrB);
  //
  // Define A/B partitioning and C accumulators
  //

  // TUTORIAL: Example of partitioning via a TiledMMA

  ThrMMA thr_mma = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K)
  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate the accumulators -- same size as the projected data
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)

  CUTE_STATIC_ASSERT_V(  shape(tCrC) ==   shape(tCgC));                // (MMA,MMA_M,MMA_N)
  CUTE_STATIC_ASSERT_V(size<1>(tCgC) == size<1>(tCsA));                // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(tCgC) == size<1>(tCsB));                // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCsA) == size<2>(tCsB));                // MMA_K

  // Clear the accumulators
  clear(tCrC);

#if 0
  if(thread0()) {
    print("  mA : "); print(  mA); print("\n");
    print("  gA : "); print(  gA); print("\n");
    print("  sA : "); print(  sA); print("\n");
    print("tAgA : "); print(tAgA); print("\n");
    print("tAsA : "); print(tAsA); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mB : "); print(  mB); print("\n");
    print("  gB : "); print(  gB); print("\n");
    print("  sB : "); print(  sB); print("\n");
    print("tBgB : "); print(tBgB); print("\n");
    print("tBsB : "); print(tBsB); print("\n");
    print("tArA : "); print(tArA); print("\n");
  }
#endif

#if 0
  if(thread0()) {
    print("  mC : "); print(  mC); print("\n");
    print("  gC : "); print(  gC); print("\n");
    print("tCsA : "); print(tCsA); print("\n");
    print("tCsB : "); print(tCsB); print("\n");
    print("tCgC : "); print(tCgC); print("\n");
    print("tCrC : "); print(tCrC); print("\n");
  }
#endif

#if 1

  // TUTORIAL: Example of an inner loop that pipelines compute with reads
  //           from global memory by staging through register and shared memory.
  //   Data is read from global to registers, then to shared via the TiledCopy partitions
  //   gemm(.) operates on the shared memory directly via the TiledMMA partitions

  auto K_TILE_MAX = size<3>(tAgA);

  for (int k_tile = 0; k_tile < K_TILE_MAX; ++k_tile)
  {
    // Copy rmem to smem with tA|tB thread-partitioned tensors
    __syncthreads();         // Wait for all threads to consume smem
    copy(tArA, tAsA);
    copy(tBrB, tBsB);
    __syncthreads();         // Wait for all threads to consume smem

    // Copy gmem to rmem for k_tile+1 with tA|tB thread-partitioned tensors
    int k_tile_next = (k_tile + 1 < K_TILE_MAX) ? k_tile + 1 : k_tile;
    copy(copy_a, tAgA(_,_,_,k_tile_next), tArA);
    copy(copy_b, tBgB(_,_,_,k_tile_next), tBrB);
    // TUTORIAL: The above call to copy(copy_a, tAgA(_,_,_,k_tile_next), tArA) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       copy_a.call(tAgA(_,m,k), tArA(_,m,k);
    //     }
    //   }

    // Compute gemm on mma-partitioned smem
    gemm(mma, tCsA, tCsB, tCrC);
    // TUTORIAL: The above call to gemm(tCsA, tCsB, tCrC) is equivalent to
    //   CUTE_UNROLL
    //   for (int k = 0; k < size<1>(tCsA); ++k) {
    //     CUTE_UNROLL
    //     for (int m = 0; m < size<0>(tCrC); ++m) {
    //       CUTE_UNROLL
    //       for (int n = 0; n < size<1>(tCrC); ++n) {
    //         mma.call(tCsA(_,m,k), tCsB(_,n,k), tCrC(_,m,n);
    //       }
    //     }
    //   }
  }

#endif

  //
  // Epilogue
  //

  axpby(alpha, tCrC, beta, tCgC);
}

// Setup params for a NT GEMM
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
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define NT strides (mixed)
  auto dA = make_stride(Int<1>{}, ldA);                      // (dM, dK)
  auto dB = make_stride(Int<1>{}, ldB);                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<32>{};
  auto bN = Int<32>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape(bM, bK));                 // (m,k) -> smem_idx; m-major
  auto sB = make_layout(make_shape(bN, bK));                 // (n,k) -> smem_idx; n-major
  auto sC = make_layout(make_shape(bM, bN));                 // (m,n) -> smem_idx; m-major

  // Define the thread layouts (static)

  // TUTORIAL: Construct TiledCopy with a particular Copy_Atom to use and
  //           define the partitioning pattern to apply.
  // Each thread will (try to) copy 4x1 elements of type TA using 128-bit copy.
  // Use 32x8 of these threads.

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TA>{},
                                    Layout<Shape<_8, _4>>{},  // Thr layout 32x8 m-major
                                    Layout<Shape<_4, _2>>{}); // Val layout  4x1 m-major
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<uint128_t>, TB>{},
                                    Layout<Shape<_8, _4>>{},  // Thr layout 32x8 n-major
                                    Layout<Shape<_4, _2>>{}); // Val layout  4x1 n-major

  // TUTORIAL: Construct TiledMMA with a particular MMA_Atom to use and
  //           define the partitioning pattern to apply.
  // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
  // Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC, TA, TB>{},
                                 Layout<Shape<_8, _4, _1>>{});  // 16x16x1 UniversalFMA

  // print_latex(mmaC);
  fprint_latex(get_file_ptr(), mmaC);

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC,
       alpha, beta);
}

// Setup params for a TN GEMM
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
  auto prob_shape = make_shape(M, N, K);                     // (M, N, K)

  // Define TN strides (mixed)
  auto dA = make_stride(ldA, Int<1>{});                      // (dM, dK)
  auto dB = make_stride(ldB, Int<1>{});                      // (dN, dK)
  auto dC = make_stride(Int<1>{}, ldC);                      // (dM, dN)

  // Define CTA tile sizes (static)
  auto bM = Int<32>{};
  auto bN = Int<8>{};
  auto bK = Int<8>{};
  auto cta_tiler = make_shape(bM, bN, bK);                   // (BLK_M, BLK_N, BLK_K)

  // Define the smem layouts (static)
  auto sA = make_layout(make_shape (      bM,          bK),
                        make_stride(Int<1>{}, bM+Int<1>{}));        // (m,k) -> smem_idx; padded m-major
  auto sB = make_layout(make_shape (      bN,          bK),
                        make_stride(Int<1>{}, bN+Int<1>{}));        // (n,k) -> smem_idx; padded n-major
  auto sC = make_layout(make_shape(bM, bN));                        // (m,n) -> smem_idx

  // TUTORIAL: Construct TiledCopy to define the Copy_Atom to use and the
  //           partitioning pattern to apply.
  // Each thread will copy 1x1 elements of type TA.
  // Use 32x8 of these threads arranged in k-major.

  TiledCopy copyA = make_tiled_copy(Copy_Atom<UniversalCopy<TA>, TA>{},
                                    Layout<Shape<_8, _4>, Stride<_4, _1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape<_1, _1>>{});              // Val layout  1x1
  TiledCopy copyB = make_tiled_copy(Copy_Atom<UniversalCopy<TB>, TB>{},
                                    Layout<Shape<_8, _4>, Stride<_4, _1>>{}, // Thr layout 32x8 k-major
                                    Layout<Shape<_1, _1>>{});              // Val layout  1x1

  // TUTORIAL: Construct TiledMMA to define the MMA_Atom to use and the
  //           partitioning pattern to apply.
  // Use a 1x1x1 FMA on the types TC += TA * TB. Each atom requires a single thread.
  // Reproduce that atom 16x16x1 times (m-major) across threads so that we use 256 threads.

  TiledMMA mmaC = make_tiled_mma(UniversalFMA<TC, TA, TB>{},
                                 Layout<Shape<_8, _4, _1>>{});  // 16x16x1 TiledMMA

#if 0
  print(copyA);
  print(copyB);
  print(mmaC);
#endif

#if 0
  print_latex(copyA);
  print_latex(copyB);
  print_latex(mmaC);
#endif

  dim3 dimBlock(size(mmaC));
  dim3 dimGrid(size(ceil_div(M, bM)),
               size(ceil_div(N, bN)));
  gemm_device<<<dimGrid, dimBlock, 0, stream>>>
      (prob_shape, cta_tiler,
       A, dA, sA, copyA,
       B, dB, sB, copyB,
       C, dC, sC, mmaC,
       alpha, beta);
}

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
test_mma_v2(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     thrust::host_vector<TA> const& h_A, int ldA,
     thrust::host_vector<TB> const& h_B, int ldB,
     Beta beta,
     thrust::host_vector<TC>      & h_C, int ldC)
{
  printf("[test_mma_v2] start\n");
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
  CUTE_CHECK_LAST();

  // Copy output matrix
  h_C = d_C;
  printf("[test_mma_v2] finish\n");
}

template void test_mma_v2<float, float, float, float, float>(char transA, char transB, int m, int n, int k,
                                                      float alpha,
                                                      thrust::host_vector<float> const& h_A, int ldA,
                                                      thrust::host_vector<float> const& h_B, int ldB,
                                                      float beta,
                                                      thrust::host_vector<float>      & h_C, int ldC);

} // namespace v2
