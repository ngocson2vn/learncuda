#pragma once

#include <thrust/host_vector.h>

#include "cutlass/pipeline/sm90_pipeline.hpp"
#include "cutlass/arch/barrier.h"
#include "cutlass/arch/mma_sm90.h"

namespace v1 {
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
test_mma_v1(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     thrust::host_vector<TA> const& h_A, int ldA,
     thrust::host_vector<TB> const& h_B, int ldB,
     Beta beta,
     thrust::host_vector<TC>      & h_C, int ldC);

} // namespace v1

namespace v2 {

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
test_mma_v2(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     thrust::host_vector<TA> const& h_A, int ldA,
     thrust::host_vector<TB> const& h_B, int ldB,
     Beta beta,
     thrust::host_vector<TC>      & h_C, int ldC);

} // namespace v2


using namespace cute;
namespace sm90 {

template <class ElementA,
          class ElementB,
          class SmemLayoutA,  // (bM,bK,bP)
          class SmemLayoutB>  // (bN,bK,bP)
struct SharedStorage
{
  array_aligned<ElementA, cosize_v<SmemLayoutA>> smem_A;
  array_aligned<ElementB, cosize_v<SmemLayoutB>> smem_B;

  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class ProblemShape, class CtaTiler,
          class TA, class SmemLayoutA, class TmaA,
          class TB, class SmemLayoutB, class TmaB,
          class TC, class CStride, class TiledMma,
          class Alpha, class Beta>
__global__ static
__launch_bounds__(decltype(size(TiledMma{}))::value)
void
gemm_device(ProblemShape shape_MNK, CtaTiler cta_tiler,
            TA const* A, CUTLASS_GRID_CONSTANT TmaA const tma_a,
            TB const* B, CUTLASS_GRID_CONSTANT TmaB const tma_b,
            TC      * C, CStride dC, TiledMma mma,
            Alpha alpha, Beta beta)
{
  // Preconditions
  CUTE_STATIC_ASSERT_V(rank(shape_MNK) == Int<3>{});                   // (M, N, K)
  CUTE_STATIC_ASSERT_V(rank(cta_tiler) == Int<3>{});                   // (BLK_M, BLK_N, BLK_K)

  static_assert(is_static<SmemLayoutA>::value);
  static_assert(is_static<SmemLayoutB>::value);

  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutA{}) == size<0>(cta_tiler));  // BLK_M
  CUTE_STATIC_ASSERT_V(size<0>(SmemLayoutB{}) == size<1>(cta_tiler));  // BLK_N
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutA{}) == size<2>(cta_tiler));  // BLK_K
  CUTE_STATIC_ASSERT_V(size<1>(SmemLayoutB{}) == size<2>(cta_tiler));  // BLK_K

  CUTE_STATIC_ASSERT_V(congruent(select<0,1>(shape_MNK), dC));         // dC strides for shape MN

  //
  // Full and Tiled Tensors
  //

  // Represent the full tensors
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M,K));                   // (M,K) TMA Tensor
  Tensor mB = tma_b.get_tma_tensor(make_shape(N,K));                   // (N,K) TMA Tensor
  Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M,N), dC);      // (M,N)

  // Get the appropriate blocks for this thread block
  auto cta_coord = make_coord(blockIdx.x, blockIdx.y, _);              // (m,n,k)
  Tensor gA = local_tile(mA, cta_tiler, cta_coord, Step<_1, X,_1>{});  // (BLK_M,BLK_K,k) k = K/bK
  Tensor gB = local_tile(mB, cta_tiler, cta_coord, Step< X,_1,_1>{});  // (BLK_N,BLK_K,k)
  Tensor gC = local_tile(mC, cta_tiler, cta_coord, Step<_1,_1, X>{});  // (BLK_M,BLK_N)

  // Shared memory tensors
  extern __shared__ char shared_memory[];

  using SharedStorage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  SharedStorage& smem = *reinterpret_cast<SharedStorage*>(shared_memory);
  if (thread0()) {
    printf("Dynamic shared memory size: %ld\n", sizeof(SharedStorage));
  }

  Tensor sA = make_tensor(make_smem_ptr(smem.smem_A.data()), SmemLayoutA{}); // (BLK_M,BLK_K,PIPE)
  Tensor sB = make_tensor(make_smem_ptr(smem.smem_B.data()), SmemLayoutB{}); // (BLK_N,BLK_K,PIPE)

  //
  // Partition the copying of A and B tiles
  //
  // TUTORIAL:
  //   These are TMA partitionings, which have a dedicated custom partitioner.
  //   The Int<0>, Layout<_1> indicates that the TMAs are not multicasted.
  //     Any multicasting must be in conformance with tma_x constructed with make_tma_atom on host.
  //   The group_modes<0,2> transforms the (X,Y,Z)-shaped tensors into ((X,Y),Z)-shaped tensors
  //     with the understanding that the TMA is responsible for everything in mode-0.
  //   The tma_partition reorders and offsets mode-0 according to the tma_x atom and the multicast info.
  //

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sA), group_modes<0,2>(gA));  // (TMA,k) and (TMA,PIPE)
  if (thread0()) {
    printf("tma_a: "); print(tma_a); printf("\n");
    printf("tAsA: "); print(tAsA); printf("\n");
  }

  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{},
                                    group_modes<0,2>(sB), group_modes<0,2>(gB));  // (TMA,k) and (TMA,PIPE)
  if (thread0()) {
    printf("tBsB: "); print(tBsB); printf("\n");
  }

  // The TMA is responsible for copying everything in mode-0 of tAsA and tBsB
  constexpr int kTmaTransactionBytes = CUTE_STATIC_V(size<0>(tAsA)) * sizeof(TA) +
                                       CUTE_STATIC_V(size<0>(tBsB)) * sizeof(TB);
  if (thread0()) {
    printf("kTmaTransactionBytes: %d\n", kTmaTransactionBytes);
  }

  //
  // PREFETCH
  //

  auto K_PIPE_MAX = size<1>(tAsA); // = bP = 3
  if (thread0()) {
    printf("blockIdx.x=%d blockIdx.y=%d K_PIPE_MAX = %ld\n", blockIdx.x, blockIdx.y, (int)K_PIPE_MAX);
  }

  // Total count of tiles
  int k_tile_count = size<1>(tAgA); // = K/bK = 1024 / 64 = 16
  if (thread0()) {
    printf("blockIdx.x=%d blockIdx.y=%d k_tile_count = %ld\n", blockIdx.x, blockIdx.y, (int)k_tile_count);
  }

  // Current tile index in gmem to read from
  int k_tile = 0;

  // Initialize Barriers
  int warp_idx = cutlass::canonical_warp_idx_sync();
  int lane_predicate = cute::elect_one_sync();
  uint64_t* producer_mbar = smem.tma_barrier;
  uint64_t* consumer_mbar = smem.mma_barrier;

  using ProducerBarType = cutlass::arch::ClusterTransactionBarrier;  // TMA
  using ConsumerBarType = cutlass::arch::ClusterBarrier;             // MMA
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe) {
    if ((warp_idx == 0) && lane_predicate) {
      ProducerBarType::init(&producer_mbar[pipe],   1);
      ConsumerBarType::init(&consumer_mbar[pipe], blockDim.x * blockDim.y * blockDim.z); // Entire block
    }
  }
  // Ensure barrier init is complete on all CTAs
  cluster_sync();

  // Start async loads for all pipes
  CUTE_UNROLL
  for (int pipe = 0; pipe < K_PIPE_MAX; ++pipe)
  {
    if ((warp_idx == 0) && lane_predicate)
    {
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
    }
    --k_tile_count;
    ++k_tile;
  }

  //
  // Define A/B partitioning and C accumulators
  //
  // TUTORIAL:
  //   The tCrA and tCrB are actually Tensors of MMA Descriptors constructed as views of SMEM.
  //   The MMA Descriptor generation is automatic via inspection and validation of the SMEM Layouts.
  //   Because the MMA reads directly from SMEM and the fragments are descriptors rather than registers,
  //     there is no need for copy(tCsA, tCrA) in the mainloop.
  //

  ThrMMA thr_mma = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr_mma.partition_A(sA);                               // (MMA,MMA_M,MMA_K,PIPE)
  /*
    tCsA: Sw<3,4,3>_smem_ptr[16b](0x148000000400) o ((_64,(_8,_2)),_2,_4,_3):((_1,(_64,_1024)),_512,_2048,_8192)
    MMA is the MxK shape of the MMA Atom SM90_64x64x16_F16F16F16_SS.
    MMA_M and MMA_K are the extents of its tiling across the M and K modes of sA (so that MMA_M=bM/64=2 and MMA_K=bK/16=4).
    PIPE is the number of stages.
  */

  Tensor tCsB = thr_mma.partition_B(sB);                               // (MMA,MMA_N,MMA_K,PIPE)
  Tensor tCgC = thr_mma.partition_C(gC);                               // (MMA,MMA_M,MMA_N)

  // Allocate accumulators and clear them
  Tensor tCrC = thr_mma.make_fragment_C(tCgC);                         // (MMA,MMA_M,MMA_N)
  clear(tCrC);

  // Allocate "fragments"
  Tensor tCrA = thr_mma.make_fragment_A(tCsA);                         // (MMA,MMA_M,MMA_K,PIPE)
  Tensor tCrB = thr_mma.make_fragment_B(tCsB);                         // (MMA,MMA_N,MMA_K,PIPE)

  if (thread0()) {
    printf("\n");
    printf("blockIdx.x=%d blockIdx.y=%d tCgC: ", blockIdx.x, blockIdx.y);
    print(tCgC);
    printf("\n");

    printf("blockIdx.x=%d blockIdx.y=%d tCsA: ", blockIdx.x, blockIdx.y);
    print(tCsA);
    printf("\n");

    printf("blockIdx.x=%d blockIdx.y=%d tCsB: ", blockIdx.x, blockIdx.y);
    print(tCsB);
    printf("\n\n");

    printf("\n");
    printf("blockIdx.x=%d blockIdx.y=%d tCrC: ", blockIdx.x, blockIdx.y);
    print(tCrC);
    printf("\n");

    printf("blockIdx.x=%d blockIdx.y=%d tCrA: ", blockIdx.x, blockIdx.y);
    print(tCrA);
    printf("\n");

    printf("blockIdx.x=%d blockIdx.y=%d tCrB: ", blockIdx.x, blockIdx.y);
    print(tCrB);
    printf("\n\n");
  }

  //
  // PIPELINED MAIN LOOP
  //
  // TUTORIAL:
  //   Rather than interleaving the stages and instructions like in SM70 and SM80,
  //     the SM90 mainloops rely on explicit producer-consumer synchronization
  //     on the purely async instructions TMA and MMA.
  //   More advanced pipeline and warp-specialization strategies are available in CUTLASS mainloops.
  //

  // A PipelineState is a circular pipe index [.index()] and a pipe phase [.phase()]
  //   that flips each cycle through K_PIPE_MAX.
  auto write_state = cutlass::PipelineState<K_PIPE_MAX>();             // TMA writes
  auto read_state  = cutlass::PipelineState<K_PIPE_MAX>();             // MMA  reads

  CUTE_NO_UNROLL
  while (k_tile_count > -K_PIPE_MAX)
  {
    // Wait for Producer to complete
    int read_pipe = read_state.index();
    ProducerBarType::wait(&producer_mbar[read_pipe], read_state.phase());

    // MMAs to cover 1 K_TILE
    warpgroup_arrive(); // wgmma.fence
    
    gemm(mma, tCrA(_,_,_,read_pipe), tCrB(_,_,_,read_pipe), tCrC);     // (V,M) x (V,N) => (V,M,N)
    /*============================================================================================
      cutlass/include/cute/algorithm/gemm.hpp:
      ============================================================
      L1 gemm runs 16 times because k_tile_count = 16
      ============================================================
      template <class MMA,
                class TA, class ALayout,
                class TB, class BLayout,
                class TC, class CLayout>
      CUTE_HOST_DEVICE
      void
      gemm(MMA_Atom<MMA>       const& mma,
          Tensor<TA, ALayout> const& A,
          Tensor<TB, BLayout> const& B,
          Tensor<TC, CLayout>      & C)
      {
        return gemm(mma, C, A, B, C);
      }

      ============================================================
      L2 gemm runs 16 times because L1 gemm calls it 16 times
      ============================================================
      // Dispatch [5]: (V,M,K) x (V,N,K) => (V,M,N)
      template <class MMA,
                class TD, class DLayout,
                class TA, class ALayout,
                class TB, class BLayout,
                class TC, class CLayout,
                __CUTE_REQUIRES(DLayout::rank == 3 && is_rmem<TD>::value &&
                                ALayout::rank == 3 && is_rmem<TA>::value &&
                                BLayout::rank == 3 && is_rmem<TB>::value &&
                                CLayout::rank == 3 && is_rmem<TC>::value)>
      CUTE_HOST_DEVICE
      void
      gemm(MMA_Atom<MMA>       const& mma,
          Tensor<TD, DLayout>      & D,  // (V,M,N) Logical data
          Tensor<TA, ALayout> const& A,  // (V,M,K) Logical data
          Tensor<TB, BLayout> const& B,  // (V,N,K) Logical data
          Tensor<TC, CLayout> const& C)  // (V,M,N) Logical data
      {
        CUTE_STATIC_ASSERT_V(size<1>(A) == size<1>(C));  // AM == CM
        CUTE_STATIC_ASSERT_V(size<1>(B) == size<2>(C));  // BN == CN
        CUTE_STATIC_ASSERT_V(size<2>(A) == size<2>(B));  // AK == BK
        CUTE_STATIC_ASSERT_V(size<0>(C) == size<0>(D) && size<1>(C) == size<1>(D) && size<2>(C) == size<2>(D));
        auto K = size<2>(A);

        CUTE_UNROLL
        for (int k = 0; k < K; ++k) {
          gemm(mma, D, A(_,_,k), B(_,_,k), C);
        }
      }

      ============================================================
      L3 gemm runs 64 times because K = size<2>(A) = MMA_K = 4
      ============================================================
      // Dispatch [4]: (V,M) x (V,N) => (V,M,N)
      template <class MMA,
                class TD, class DLayout,
                class TA, class ALayout,
                class TB, class BLayout,
                class TC, class CLayout,
                __CUTE_REQUIRES(DLayout::rank == 3 && is_rmem<TD>::value &&
                                ALayout::rank == 2 && is_rmem<TA>::value &&
                                BLayout::rank == 2 && is_rmem<TB>::value &&
                                CLayout::rank == 3 && is_rmem<TC>::value)>
      CUTE_HOST_DEVICE
      void
      gemm(MMA_Atom<MMA>       const& mma,
          Tensor<TD, DLayout>      & D,  // (V,M,N) Logical data
          Tensor<TA, ALayout> const& A,  // (V,M)   Logical data
          Tensor<TB, BLayout> const& B,  // (V,N)   Logical data
          Tensor<TC, CLayout> const& C)  // (V,M,N) Logical data
      {
        ...
      }

      ============================================================
      L4 gemm runs 256 times because L3 gemm calls it 64 * 4 times
      ============================================================
      // Dispatch [1]: (V) x (V) => (V)
      template <class MMA,
                class TD, class DLayout,
                class TA, class ALayout,
                class TB, class BLayout,
                class TC, class CLayout,
                __CUTE_REQUIRES(DLayout::rank == 1 && is_rmem<TD>::value &&
                                ALayout::rank == 1 && is_rmem<TA>::value &&
                                BLayout::rank == 1 && is_rmem<TB>::value &&
                                CLayout::rank == 1 && is_rmem<TC>::value)>
      CUTE_HOST_DEVICE
      void
      gemm(MMA_Atom<MMA>       const& mma,
          Tensor<TD, DLayout>      & D,  // (V) Logical data
          Tensor<TA, ALayout> const& A,  // (V) Logical data
          Tensor<TB, BLayout> const& B,  // (V) Logical data
          Tensor<TC, CLayout> const& C)  // (V) Logical data
      {
        // No static assertions on (V), MMA checks compatibility
        mma.call(D, A, B, C);
      }
    ============================================================================================*/

    warpgroup_commit_batch();

    // Wait for all MMAs in a K_TILE to complete
    warpgroup_wait<0>();

    // Notify that consumption is done
    ConsumerBarType::arrive(&consumer_mbar[read_pipe]);
    ++read_state;

    if ((warp_idx == 0) && lane_predicate)
    {
      int pipe = write_state.index();
      // Wait for Consumer to complete consumption
      ConsumerBarType::wait(&consumer_mbar[pipe], write_state.phase());
      // Set expected Tx Bytes after each reset / init
      ProducerBarType::arrive_and_expect_tx(&producer_mbar[pipe], kTmaTransactionBytes);
      copy(tma_a.with(producer_mbar[pipe]), tAgA(_,k_tile), tAsA(_,pipe));
      copy(tma_b.with(producer_mbar[pipe]), tBgB(_,k_tile), tBsB(_,pipe));
      ++write_state;
    }
    --k_tile_count;
    ++k_tile;
  }

  //
  // Epilogue (unpredicated)
  //

  axpby(alpha, tCrC, beta, tCgC);
}

} // namespace sm90
