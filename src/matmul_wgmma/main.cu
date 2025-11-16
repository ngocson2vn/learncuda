#include <cstdio>
#include <bitset>
#include <random>

#include "device.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "fprint_mat.h"

using bf16_t = nv_bfloat16;

#define STRINGIFY_TYPE(type) #type
#define TYPE_TO_STR(type) STRINGIFY_TYPE(type)


__device__ void warpgroup_arrive()
{
  asm volatile ("wgmma.fence.sync.aligned;\n" ::: "memory");
}

// Marks the commit point for one or more sized batch of warpgroup MMAs.
__device__ void warpgroup_commit_batch()
{
  asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
}

template <int N>
__device__ void warpgroup_wait()
{
  static_assert(N >= 0 && N <= 7, "WGMMA wait: N must be in range [0, 7]");
  asm volatile("wgmma.wait_group.sync.aligned %0;\n" :: "n"(N) : "memory");
}

__device__ inline void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
}

__device__ void mbarrier_init(void const* smem_ptr, uint32_t arrive_count)
{
  uint64_t smem_addr = __cvta_generic_to_shared(smem_ptr);
  asm volatile(
    "{\n\t"
      "mbarrier.init.shared::cta.b64 [%0], %1; \n"
    "}"
    :
    : "l"(smem_addr), "r"(arrive_count)
  );
}

// Performs an arrive operation + expected transaction count
__device__ void mbarrier_arrive_and_expect_tx_bytes(void const* smem_ptr, uint32_t bytes) {
  uint64_t smem_addr = __cvta_generic_to_shared(smem_ptr);
  asm volatile(
    "{\n\t"
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1; \n\t"
    "}"
    :
    : "l"(smem_addr), "r"(bytes)
  );
}

__device__ void mbarrier_wait(void const* smem_ptr, uint32_t phaseParity) {
  uint64_t smem_addr = __cvta_generic_to_shared(smem_ptr);
  // Arbitrarily large timer value after which try-wait expires and re-tries.
  uint32_t ticks = 0x989680;
  asm volatile(
    "{\n\t"
      ".reg .pred       P1; \n\t"
      "LAB_WAIT: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1, %2; \n\t"
        "@P1 bra DONE; \n\t"
        "bra     LAB_WAIT; \n\t"
      "DONE: \n"
    "}"
    :
    : "l"(smem_addr), "r"(phaseParity), "r"(ticks)
  );
}

__device__ inline void cp_async_bulk_tensor_2d_global_to_shared(void* dst, const void* tensorMap, int col, int row, void* smem_bar) {
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
         :
         : "l"(__cvta_generic_to_shared(dst)),
           "l"(reinterpret_cast<uint64_t>(tensorMap)),
           "r"(col),
           "r"(row),
           "l"(__cvta_generic_to_shared(smem_bar))
         : "memory"
    );
}

__device__ inline void cp_async_bulk_tensor_2d_shared_to_global(const void* tensorMap, int col, int row, const void* smem_ptr) {
    asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
         :
         : "l"(reinterpret_cast<uint64_t>(tensorMap)),
           "r"(col), 
           "r"(row),
           "l"(__cvta_generic_to_shared(smem_ptr))
         : "memory"
    );
}

__device__ inline void cp_async_bulk_commit_group() {
  asm volatile("cp.async.bulk.commit_group;" : : :);
}

template <int N>
__device__ inline void cp_async_bulk_wait_group_read() {
  asm volatile("cp.async.bulk.wait_group.read %0;" : : "n"(N) : "memory");
}


template <int32_t scaleD = 1, int32_t scaleA = 1, int32_t scaleB = 1, int32_t tnspA = 0, int32_t tnspB = 0>
struct MMA_64x64x16_F32F16F16_SS
{
  __device__ static void
  fma(
      float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
      float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
      float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
      float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
      uint64_t const& desc_A,
      uint64_t const& desc_B)
  {
    asm volatile(
      "{\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{ %0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
        "  %8,  %9, %10, %11, %12, %13, %14, %15,  "
        " %16, %17, %18, %19, %20, %21, %22, %23,  "
        " %24, %25, %26, %27, %28, %29, %30, %31 },"
        " %32,"
        " %33,"
        " %34, %35, %36, %37, %38;\n"
      "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03), "+f"(d04), "+f"(d05), "+f"(d06), "+f"(d07),
          "+f"(d08), "+f"(d09), "+f"(d10), "+f"(d11), "+f"(d12), "+f"(d13), "+f"(d14), "+f"(d15),
          "+f"(d16), "+f"(d17), "+f"(d18), "+f"(d19), "+f"(d20), "+f"(d21), "+f"(d22), "+f"(d23),
          "+f"(d24), "+f"(d25), "+f"(d26), "+f"(d27), "+f"(d28), "+f"(d29), "+f"(d30), "+f"(d31)
        :  "l"(desc_A), "l"(desc_B),
           "n"(scaleD), "n"(scaleA), "n"(scaleB), "n"(tnspA), "n"(tnspB)
    );
  }
};


template<typename DataType, int ROWS, int COLS, int BOX_ROWS, int BOX_COLS, bool rowMajor = true>
bool create_tensor_map(CUtensorMap* tensorMap, void* globalAddress) {
  // tensorRank is the number of dimensions of the array.
  constexpr uint32_t tensorRank = 2;

  // In CUDA, 
  // dim0 = column
  // dim1 = row
  uint64_t globalDim[tensorRank] = {COLS, ROWS};
  if constexpr(!rowMajor) {
    globalDim[0] = ROWS;
    globalDim[1] = COLS;
  }

  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t globalStrides[tensorRank - 1] = {globalDim[0] * sizeof(DataType)};
  printf("globalStrides[0] = %ld\n", globalStrides[0]);

  // The boxDim is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t boxDim[tensorRank] = {BOX_COLS, BOX_ROWS};
  if constexpr(!rowMajor) {
    boxDim[0] = BOX_ROWS;
    boxDim[1] = BOX_COLS;
  }

  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elementStrides[tensorRank] = {1, 1};

  CUtensorMapDataType tensorDataType;
  if constexpr(std::is_same<DataType, int>()) {
    tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32;
  } else if constexpr(std::is_same<DataType, half>()) {
    tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr(std::is_same<DataType, bf16_t>()) {
    tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if constexpr(std::is_same<DataType, float>()) {
    tensorDataType = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    fprintf(stderr, "%s:%d: create_tensor_map does not support the provided DataType.\n", __FILE__, __LINE__);
    return false;
  }

  printf("CUtensorMapDataType: %d\n", tensorDataType);

  // Create the tensor descriptor.
  CUresult ret = cuTensorMapEncodeTiled(
    tensorMap,                  // CUtensorMap *tensorMap,
    tensorDataType,             // CUtensorMapDataType tensorDataType
    tensorRank,                 // cuuint32_t tensorRank,
    globalAddress,              // void *globalAddress,
    globalDim,                  // const cuuint64_t *globalDim,
    globalStrides,              // const cuuint64_t *globalStrides,
    boxDim,                     // const cuuint32_t *boxDim,
    elementStrides,             // const cuuint32_t *elementStrides,

    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,

    // Swizzling can be used to avoid shared memory bank conflicts.
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,

    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,

    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, "Failed to create CUtensorMap object");
    const char* errName;
    cuGetErrorName(ret, &errName);
    const char* errMsg;
    cuGetErrorString(ret, &errMsg);
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, errName, errMsg);
    fflush(stderr);
    return false;
  }

  return true;
}


// Ref: https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#shared-memory-matrix-layout
// Matrices in shared memory are organized into a number of smaller matrices called core matrices. 
// Each core matrix has 8 rows or columns and the size of each row is 16 bytes. 
// The core matrices occupy contiguous space in shared memory.
// Matrix A is made up of 8x2 core matrices and Matrix B is made up of 2x(N/8) core matrices.
// Core matrix A(8x8): Each row is made up of eight .f16/ .bf16 elements.
// Core matrix B(8x8): Each column is made up of eight .f16/ .bf16 elements.
template<typename DataType>
__device__ uint64_t create_matrix_desc(void* smem_ptr) {
  uint64_t desc = 0;

  // matrix-descriptor-encode(x)
  auto mde = [](uint64_t x) -> uint64_t { return ((x & 0x3FFFF) >> 0x4); };

  uint64_t smem_addr = mde(__cvta_generic_to_shared(smem_ptr));

  // Matrix start address: bits 13-0
  desc = desc | smem_addr;

  // Leading dimension byte offset: bits 29–16
  // Leading dimension byte offset of matrix A or B is the distance, in bytes, between two adjacent core matrices in the K dimension.
  uint64_t ld_byte_offset = mde(8 * 8 * sizeof(DataType)) << 16;
  desc = desc | ld_byte_offset;

  // Stride dimension byte offset: bits 45–32
  // Stride dimension byte offset of matrix A or B is the distance, in bytes, between two adjacent core matrices in the M or N dimension.
  uint64_t sd_byte_offset = mde(2 * (8 * 8 * sizeof(DataType))) << 32;
  desc = desc | sd_byte_offset;

  return desc;
}

// https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-wgmma-mma-async-m64nnk16
__device__ void gen_indices(int* indexes_tile_D, const int* shape_tile_D, const int* stride_tile_D, int thread_idx) {
  int baseRow = (thread_idx / 4) + (thread_idx / 32) * 8;
  int baseCol  = (thread_idx % 4) * 2;
  int k = 0;
  int idx = 0;
  for (int m = 0; m < (shape_tile_D[0] / 32); m++) {
    int fragment_idx = k;
    for (int n = 0; n < (shape_tile_D[1] / 8); n++) {
      idx = (baseRow + m * 8) * stride_tile_D[0] + (baseCol + n * 8) * stride_tile_D[1];
      indexes_tile_D[fragment_idx]   = idx;
      indexes_tile_D[fragment_idx+1] = idx + 1;
      fragment_idx += 4;
    }

    k += 2;
  }
}

/*===---------------------------------------------------------------------------------------------------------------===*/
// gemm_tma_wgmma_bf16_fp32
/*===---------------------------------------------------------------------------------------------------------------===*/
// Kernel uses TMA mbarriers (64B) and smem
template <typename ABType, typename DType, int M, int N, int K, int M_TILE, int N_TILE, int K_TILE>
__global__ void gemm_tma_wgmma_bf16_fp32(
  const __grid_constant__ CUtensorMap tensor_map_A,
  const __grid_constant__ CUtensorMap tensor_map_B,
  const __grid_constant__ CUtensorMap tensor_map_D
) {
#if __CUDA_ARCH__ < 900
  return; // requires SM90
#endif

  // Double buffers double_tile_A[2 * M_TILE * K_TILE]
  constexpr uint32_t size_tile_A = M_TILE * K_TILE;
  constexpr uint32_t bytes_tile_A = size_tile_A * sizeof(ABType);

  // IMPORTANT NOTE:
  // alignas(128) must precede ABType
  __shared__ alignas(128) ABType double_tile_A[2 * size_tile_A];

  // double_tile_B[2 * N_TILE * K_TILE]
  constexpr uint32_t size_tile_B = K_TILE * N_TILE;
  constexpr uint32_t bytes_tile_B = size_tile_B * sizeof(ABType);
  __shared__ alignas(128) ABType double_tile_B[2 * size_tile_B];

  static_assert(bytes_tile_A + bytes_tile_B < 49152, "SMEM size exceeds the maximum value of 49152 bytes");

  // tile_D[M_TILE * N_TILE]
  constexpr int size_tile_D = M_TILE * N_TILE;
  constexpr int shape_tile_D[] = {M_TILE, N_TILE};
  constexpr int stride_tile_D[] = {N_TILE, 1};
  __shared__ alignas(128) DType tile_D[size_tile_D];


  // two mbarriers: one for A, one for B
  // Ref: https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment
  __shared__ uint64_t mbar_A;
  __shared__ uint64_t mbar_B;

  // Accumulators for each thread
  DType acc[32];
  #pragma unroll
  for (int i = 0; i < 32; ++i) {
    acc[i] = DType(0);
  }

  // Compute block tile indices
  int block_m = blockIdx.y; // y is row
  int block_n = blockIdx.x; // x is col
  int m0 = block_m * M_TILE;
  int n0 = block_n * N_TILE;

  // Bounds check for full tiles (simple version)
  if (m0 + M_TILE > M || n0 + N_TILE > N) {
    return;
  }

  auto tma_load_tile_A = [&](int k_tile, int stage) {
    void* ptr_tile_A = double_tile_A + stage * size_tile_A;
    int col_A = k_tile;
    int row_A = m0;
    cp_async_bulk_tensor_2d_global_to_shared(
      ptr_tile_A,
      &tensor_map_A,
      col_A, // col goes first
      row_A, // row goes next
      &mbar_A
    );
  };

  auto tma_load_tile_B = [&](int k_tile, int stage) {
    void* ptr_tile_B = double_tile_B + stage * size_tile_B;
    int col_B = n0;
    int row_B = k_tile;
    cp_async_bulk_tensor_2d_global_to_shared(
      ptr_tile_B,
      &tensor_map_B,
      col_B, // col goes first
      row_B, // row goes next
      &mbar_B
    );
  };

  // // Form warp and lane info
  int tid = threadIdx.x;

  // Pipeline: prefetch stage 0
  int stage = 0;

  // Initialize mbarriers once per block
  if (tid == 0) {
    // mbarrier.init.shared::cta [addr], expected_tx;
    mbarrier_init(&mbar_A, /*arrive_count=*/1); // NOTE: current phase = 0
    mbarrier_init(&mbar_B, /*arrive_count=*/1); // NOTE: current phase = 0

    // Arrive to mbarriers for first loads
    printf("bytes_tile_A: %u\n", bytes_tile_A);
    mbarrier_arrive_and_expect_tx_bytes(&mbar_A, bytes_tile_A);

    printf("bytes_tile_B: %u\n", bytes_tile_B);
    mbarrier_arrive_and_expect_tx_bytes(&mbar_B, bytes_tile_B);

    // Load tile_A
    tma_load_tile_A(/*k_tile=*/0, stage);

    // Load tile_B
    tma_load_tile_B(/*k_tile=*/0, stage);
  }

  // Syncthreads so initialized mbarriers are visible to all threads.
  __syncthreads();
  // printf("threadIdx.x %d synced\n", threadIdx.x);

  // Wait for both tiles to be ready before first compute
  // NOTE: after TMA transactions have finished, both current phases will be advanced to 1
  mbarrier_wait(&mbar_A, /*phaseParity=*/stage);
  mbarrier_wait(&mbar_B, /*phaseParity=*/stage);
  // printf("threadIdx.x %d arrived\n", threadIdx.x);

  // For debugging with only 1 block
#if 1
  if (tid == 0) {
    int idx = 0;
    printf("tile_A: M_TILE: %d, K_TILE: %d\n", M_TILE, K_TILE);
    for (int m = 0; m < M_TILE; m++) {
      printf("m = %2d: ", m);
      for (int k = 0; k < K_TILE; k++) {
        idx = m * K_TILE + k;
        printf("%6.2f", (float)double_tile_A[idx]);
      }
      printf("\n");
    }

    printf("\n\n");

    printf("tile_B: K_TILE: %d, N_TILE: %d\n", K_TILE, N_TILE);
    for (int k = 0; k < K_TILE; k++) {
      printf("k = %2d: ", k);
      for (int n = 0; n < N_TILE; n++) {
        idx = k + n * K_TILE;
        printf("%6.2f", (float)double_tile_B[idx]);
      }
      printf("\n");
    }
  }
#endif

  // Create a MMA atom
  MMA_64x64x16_F32F16F16_SS mma_atom;

  // Main K loop with double buffering
  for (int k0 = 0; k0 < K; k0 += K_TILE) {
    int next_k = k0 + K_TILE;
    int next_stage = stage ^ 1;

    // Launch next TMA loads if there is a next slice
    if (next_k < K) {
      if (tid == 0) {
        mbarrier_arrive_and_expect_tx_bytes(&mbar_A, bytes_tile_A);
        tma_load_tile_A(next_k, next_stage);

        mbarrier_arrive_and_expect_tx_bytes(&mbar_B, bytes_tile_B);
        tma_load_tile_B(next_k, next_stage);
      }
    }

    // Create matrix descriptors
    void* ptr_tile_A = double_tile_A + stage * size_tile_A;
    uint64_t desc_A = create_matrix_desc<ABType>(ptr_tile_A);

    void* ptr_tile_B = double_tile_B + stage * size_tile_B;
    uint64_t desc_B = create_matrix_desc<ABType>(ptr_tile_B);

    // Enforce an ordering of register accesses between wgmma.mma_async and other operations.
    // `wgmma.fence` instruction establishes an ordering between prior accesses to any warpgroup registers 
    // and subsequent accesses to the same registers by a `wgmma.mma_async` instruction. 
    // Only the accumulator register and the input registers containing the fragments of matrix A require this ordering.
    warpgroup_arrive();

    // Issue WGMMA operation
    mma_atom.fma(
      acc[0],  acc[1],  acc[2],  acc[3],  acc[4],  acc[5],  acc[6],  acc[7],
      acc[8],  acc[9],  acc[10], acc[11], acc[12], acc[13], acc[14], acc[15],
      acc[16], acc[17], acc[18], acc[19], acc[20], acc[21], acc[22], acc[23],
      acc[24], acc[25], acc[26], acc[27], acc[28], acc[29], acc[30], acc[31],
      desc_A,
      desc_B
    );

    warpgroup_commit_batch();

    // Wait for MMA op to complete
    warpgroup_wait<0>();

    // printf("tid %d wgmma done\n", threadIdx.x);

    // If next loads exist, wait for them to complete before swapping stage
    if (next_k < K) {
      mbarrier_wait(&mbar_A, /*phaseParity=*/next_stage); // NOTE: current phase LSB = 1
      mbarrier_wait(&mbar_B, /*phaseParity=*/next_stage); // NOTE: current phase LSB = 1
      stage = next_stage;
    }

    // printf("tid %d mbarrier_wait done\n", threadIdx.x);
  }

  // Epilogue: write acc to tile_D
  // Map acc indices -> tile D indices
  int indices_tile_D[32];
  gen_indices(indices_tile_D, shape_tile_D, stride_tile_D, tid);

  // Copy fragments stored in acc to tile_D
  // Each thread contributes 32 fragments
  for (int i = 0; i < 32; ++i) {
    int idx = indices_tile_D[i];
    if (tid == 0) {
      printf("i=%d -> idx=%d\n", i, idx);
    }
    tile_D[idx] = acc[i];
  }

  fence_proxy_async();

  // Ensure that writes by all threads are visible to TMA engine.
  __syncthreads();

  // Finally, copy tile_D back to D in global memory
  if (tid == 0) {
    int col_D = n0;
    int row_D = m0;
    cp_async_bulk_tensor_2d_shared_to_global(
      &tensor_map_D,
      col_D,
      row_D,
      tile_D
    );

    // Ref1: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#completion-mechanisms-for-asynchronous-copy-operations
    // Ref2: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
    // Create a "bulk async-group" out of the previous bulk copy operation.
    // Purpose: for tracking the completion of this group
    cp_async_bulk_commit_group();

    // Wait for all the asynchronous operations belonging to the bulk async-group are complete.
    // `0` means the executing thread waits on all the prior bulk async-groups to complete.
    cp_async_bulk_wait_group_read<0>();
  }
}
/*===---------------------------------------------------------------------------------------------------------------===*/


// Dmn = sum_k(Amk * Bkn) = sum_k(Amk * tBnk) = sum_k(A[m * K + k] * tB[n * K + k])
// where, tB is transposed B
// MatB must be pre-transposed as (N, K):(K, 1)
template <typename ABType, typename DType, int M, int N, int K>
void matmul_cpu(ABType* MatA, ABType* MatB, DType* MatD) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        MatD[m * N + n] += static_cast<DType>(MatA[m * K + k]) * static_cast<DType>(MatB[k + n * K]);
      }
    }
  }
}



/*===---------------------------------------------------------------------------------------------------------------===*/
// main
/*===---------------------------------------------------------------------------------------------------------------===*/
int main(int argc, char** argv) {
  using DType = float;
  using ABType = bf16_t;
  // Threadblock: 128 threads (4 warps) for one warp-group
  constexpr int THREADS_PER_BLOCK = 128;

  constexpr int M_TILE = 64;
  constexpr int N_TILE = 64;
  constexpr int K_TILE = 16;
  constexpr int M = M_TILE;
  constexpr int N = N_TILE;
  constexpr int K = K_TILE;

  int shape_a[] = {M, K};
  int stride_a[] = {K, 1};
  thrust::host_vector<ABType> h_A(M * K);

  int shape_b[] = {K, N};
  int stride_b[] = {1, K};
  thrust::host_vector<ABType> h_B(K * N);

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> dist(1, 10);


  for (int m = 0; m < M; m++) {
    for (int k = 0; k < K; k++) {
      h_A[m * K + k] = ABType(dist(gen));
    }
  }

  for (int k = 0; k < K; k++) {
    for (int n = 0; n < N; n++) {
      h_B[k + n * K] = ABType(dist(gen));
    }
  }

  FILE* file_ptr = get_file_ptr("output.txt");
  const char indices_A[] = {'m', 'k'};
  fprint_mat(file_ptr, "h_A", h_A.data(), indices_A, shape_a, stride_a);
  fprintf(file_ptr, "\n\n");

  const char indices_B[] = {'k', 'n'};
  fprint_mat(file_ptr, "h_B", h_B.data(), indices_B, shape_b, stride_b);
  fprintf(file_ptr, "\n\n");

  int shape_d[] = {M, N};
  int stride_d[] = {N, 1};
  thrust::host_vector<DType> h_D_cpu(M * N, DType(0));

  matmul_cpu<ABType, DType, M, N, K>(h_A.data(), h_B.data(), h_D_cpu.data());

  const char indices_D[] = {'m', 'n'};
  fprint_mat(file_ptr, "h_D_cpu", h_D_cpu.data(), indices_D, shape_d, stride_d);
  fprintf(file_ptr, "\n\n");

  // Init cuda
  device_init(0);
  printf("\n");

  // Transfer data from host to device
  thrust::device_vector<ABType> d_A = h_A;
  auto d_A_ptr = d_A.data().get();

  thrust::device_vector<ABType> d_B = h_B;
  auto d_B_ptr = d_B.data().get();

  thrust::device_vector<DType> d_D(M * N, DType(0));
  auto d_D_ptr = d_D.data().get();

  // Create tensor maps
  CUtensorMap tensor_map_A{};
  bool status_A = create_tensor_map<ABType, M, K, M_TILE, K_TILE, true>(&tensor_map_A, d_A_ptr);
  if (!status_A) {
    return EXIT_FAILURE;
  }

  CUtensorMap tensor_map_B{};
  bool status_B = create_tensor_map<ABType, K, N, K_TILE, N_TILE, false>(&tensor_map_B, d_B_ptr);
  if (!status_B) {
    return EXIT_FAILURE;
  }

  CUtensorMap tensor_map_D{};
  bool status_D = create_tensor_map<DType, M, N, M_TILE, N_TILE, true>(&tensor_map_D, d_D_ptr);
  if (!status_D) {
    return EXIT_FAILURE;
  }

  // Create a stream
  cudaStream_t stream;
  CUDA_CHECK_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Kernel invocation with runtime cluster size
  {
    cudaLaunchConfig_t config = {0};
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 1; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    // The grid dimension is not affected by cluster launch, and is still enumerated using number of blocks.
    // The grid dimension should be a multiple of cluster size.
    int gridX = M / M_TILE;
    int gridY = N / N_TILE;
    // config.gridDim = dim3(1, 1, 1); // For debugging
    config.gridDim = dim3(gridX, gridY, 1);
    config.blockDim = dim3(THREADS_PER_BLOCK);
    config.stream = stream;

    // Launch wgmma_kernel
    CUDA_CHECK_ERROR(
      cudaLaunchKernelEx(
        &config,
        gemm_tma_wgmma_bf16_fp32<ABType, DType, M, N, K, M_TILE, N_TILE, K_TILE>,
        tensor_map_A,
        tensor_map_B,
        tensor_map_D
      )
    );

    printf("Launched gemm_tma_wgmma_bf16_fp32\n");
  }

  // Copy output matrix
  DType* h_D_gpu = new DType[M * N];
  cudaMemcpyAsync(h_D_gpu, d_D_ptr, M * N * sizeof(DType), cudaMemcpyDeviceToHost, stream);

  // printf("Synchronize device start\n");
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  printf("\nSynchronize device done\n");

  fprint_mat(file_ptr, "h_D_gpu", h_D_gpu, indices_D, shape_d, stride_d);

  // Verify
  int ok = 0;
  int ng = 0;
  for (int i = 0; i < M * N; i++) {
    if (h_D_cpu[i] == h_D_gpu[i]) {
      ok++;
    } else {
      ng++;
    }
  }
  printf("D: %d ok, %d ng\n", ok, ng);
}
