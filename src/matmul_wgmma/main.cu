#include <cstdio>
#include <bitset>
#include <random>
#include <mutex>
#include <atomic>
#include <vector>
#include <thread>

#include "device.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "fprint_mat.h"

using bf16_t = nv_bfloat16;

// #define DEBUG

static std::mutex log_mutex;
#if defined(DEBUG)
#define LOG_DEBUG(format_str, ...)            \
do {                                          \
  std::lock_guard<std::mutex> lk(log_mutex);  \
  printf("DEBUG ");                           \
  printf(format_str, ##__VA_ARGS__);          \
} while(0)
#else
#define LOG_DEBUG(format_str, ...)
#endif

#define LOG_ERROR(format_str, ...)            \
do {                                          \
  std::lock_guard<std::mutex> lk(log_mutex);  \
  printf("ERROR ");                           \
  printf(format_str, ##__VA_ARGS__);          \
} while(0)


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


template<typename DataType, int swizzleMode = 0>
bool create_tensor_map(CUtensorMap* tensorMap, void* globalAddress, unsigned int ROWS, unsigned int COLS, unsigned int BOX_ROWS, unsigned int BOX_COLS) {
  // tensorRank is the number of dimensions of the array.
  constexpr uint32_t tensorRank = 2;

  // In CUDA, 
  // dim0 = column
  // dim1 = row
  uint64_t globalDim[tensorRank] = {COLS, ROWS};

  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t globalStrides[tensorRank - 1] = {COLS * sizeof(DataType)};
  LOG_DEBUG("globalStrides[0] = %ld\n", globalStrides[0]);

  // The boxDim is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t boxDim[tensorRank] = {BOX_COLS, BOX_ROWS};

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

  LOG_DEBUG("CUtensorMapDataType: %d\n", tensorDataType);

  CUtensorMapSwizzle swizzle;
  if constexpr(swizzleMode == 0) {
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
  } else if constexpr(swizzleMode == 1) {
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
  } else if constexpr(swizzleMode == 2) {
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
  } else if constexpr(swizzleMode == 3) {
    LOG_DEBUG("swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B\n");
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
  } else {
    fprintf(stderr, "%s:%d: Unsupported swizzle mode %d", __FILE__, __LINE__, swizzleMode);
    return false;
  }

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
    swizzle,

    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,

    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  if (ret != CUDA_SUCCESS) {
    fprintf(stderr, 
            "%s:%d: Failed to create CUtensorMap object: ROWS=%d COLS=%d BOX_ROWS=%d BOX_COLS=%d\n",
            __FILE__, __LINE__, ROWS, COLS, BOX_ROWS, BOX_COLS);
    const char* errName;
    cuGetErrorName(ret, &errName);
    const char* errMsg;
    cuGetErrorString(ret, &errMsg);
    fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, errName, errMsg);
    fflush(stderr);
    return false;
  }

  LOG_DEBUG("Successfully created CUtensorMap object: ROWS=%d COLS=%d BOX_ROWS=%d BOX_COLS=%d\n",
            ROWS, COLS, BOX_ROWS, BOX_COLS);
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

  uint64_t smem_addr = mde(static_cast<uint64_t>(__cvta_generic_to_shared(smem_ptr)));

  // Matrix start address: bits 13-0
  desc = desc | smem_addr;

  // Leading dimension byte offset: bits 29–16
  uint64_t ld_byte_offset = mde(8 * 8 * sizeof(DataType)) << 16;
  desc = desc | ld_byte_offset;

  // Stride dimension byte offset: bits 45–32
  uint64_t sd_byte_offset = mde(2 * 8 * 8 * sizeof(DataType)) << 32;
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
template <typename ABType, typename DType, int M_TILE, int N_TILE, int K_TILE>
__global__ void gemm_tma_wgmma_bf16_fp32(
  const __grid_constant__ CUtensorMap tensor_map_A,
  const __grid_constant__ CUtensorMap tensor_map_B,
  const __grid_constant__ CUtensorMap tensor_map_D,
  int M, int N, int K
) {
#if __CUDA_ARCH__ < 900
  return; // requires SM90
#endif

  // Double buffers double_tile_A[M_TILE * K_TILE] and tile_A[M_TILE * K_TILE]
  constexpr uint32_t size_tile_A = M_TILE * K_TILE;
  constexpr uint32_t bytes_tile_A = size_tile_A * sizeof(ABType);

  // IMPORTANT NOTE:
  // alignas(128) must precede ABType
  __shared__ alignas(128) ABType double_tile_A[2 * size_tile_A];
  __shared__ alignas(128) ABType tile_A[size_tile_A];

  // Double buffers double_tile_B[K_TILE * N_TILE] and tile_B[K_TILE * N_TILE]
  constexpr uint32_t size_tile_B = K_TILE * N_TILE;
  constexpr uint32_t bytes_tile_B = size_tile_B * sizeof(ABType);
  __shared__ alignas(128) ABType double_tile_B[2 * size_tile_B];
  __shared__ alignas(128) ABType tile_B[size_tile_B];

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

  // Create matrix descriptors
  uint64_t desc_A = create_matrix_desc<ABType>(tile_A);
  uint64_t desc_B = create_matrix_desc<ABType>(tile_B);

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

  auto stage_tile_A = [&size_tile_A](int stage) -> ABType* {
    return double_tile_A + stage * size_tile_A;
  };

  auto stage_tile_B = [&size_tile_B](int stage) -> ABType* {
    return double_tile_B + stage * size_tile_B;
  };

  auto tma_load_tile_A = [&m0, &tensor_map_A](ABType* buffer_tile_A, int k_tile) {
    int col_A = k_tile;
    int row_A = m0;
    cp_async_bulk_tensor_2d_global_to_shared(
      buffer_tile_A,
      &tensor_map_A,
      col_A, // col goes first
      row_A, // row goes next
      &mbar_A
    );
  };

  auto tma_load_tile_B = [&n0, &tensor_map_B](ABType* buffer_tile_B, int k_tile) {
    int col_B = n0;
    int row_B = k_tile;
    cp_async_bulk_tensor_2d_global_to_shared(
      buffer_tile_B,
      &tensor_map_B,
      col_B, // col goes first
      row_B, // row goes next
      &mbar_B
    );
  };

  auto copy_tile_A = [&](ABType* buffer_tile_A, int tid) {
    int m = tid / 2;
    int k_start = 8 * (tid % 2);
    for (int i = 0; i < 8; i++) {
      int k = k_start + i;
      int idx = m * K_TILE + k;
      int linear_idx = m * 8 + (k % 8) + (m / 8) * 64 + (k / 8) * 64;
      tile_A[linear_idx] = buffer_tile_A[idx];
    }
  };

  auto copy_tile_B = [&](ABType* buffer_tile_B, int tid) {
    int k_start = 8 * (tid % 2);
    int n = tid / 2;
    for (int i = 0; i < 8; i++) {
      int k = k_start + i;
      int idx = k * N_TILE + n;
      int linear_idx = n * 8 + (k % 8) + (n / 8) * 64 + (k / 8) * 64;
      tile_B[linear_idx] = buffer_tile_B[idx];
    }
  };

  // Thread ID
  int tid = threadIdx.x;

  // Pipeline: prefetch stage 0
  int stage = 0;

  // Initialize mbarriers once per block
  if (tid == 0) {
    // mbarrier.init.shared::cta [addr], expected_tx;
    mbarrier_init(&mbar_A, /*arrive_count=*/1); // NOTE: current phase = 0
    mbarrier_init(&mbar_B, /*arrive_count=*/1); // NOTE: current phase = 0

    // Arrive to mbarriers for first loads
    mbarrier_arrive_and_expect_tx_bytes(&mbar_A, bytes_tile_A);
    mbarrier_arrive_and_expect_tx_bytes(&mbar_B, bytes_tile_B);

    // 
    // NOTE: TMA loads are asynchronous proxy operations.
    // They write into double_tile_A/B in the current stage.
    // 

    // Load tile_A
    tma_load_tile_A(stage_tile_A(stage), /*k_tile=*/0);

    // Load tile_B
    tma_load_tile_B(stage_tile_B(stage), /*k_tile=*/0);
  }

  // Syncthreads so initialized mbarriers are visible to all threads.
  __syncthreads();

  // Wait for both tiles to be ready before first compute
  // NOTE: after TMA transactions have finished, both mbar's current phases will be advanced to 1
  // The mbarrier_wait() only guarantees that the transaction count has been satisfied and 
  // the data is in shared memory somewhere — but not necessarily visible to all threads in the CTA yet.
  mbarrier_wait(&mbar_A, /*phaseParity=*/stage);
  mbarrier_wait(&mbar_B, /*phaseParity=*/stage);

  // This fence makes the loaded data globally visible to the CTA.
  // Only after this fence are the normal stores in copy_tile_A/B guaranteed 
  // to read the up-to-date data from double_tile_A/B.
  fence_proxy_async();

  // Create a MMA atom
  MMA_64x64x16_F32F16F16_SS mma_atom;

  // Main K loop with double buffering
  for (int k0 = 0; k0 < K; k0 += K_TILE) {
    int next_k = k0 + K_TILE;
    int next_stage = stage ^ 1;

    // Launch next TMA loads if there is a next slice
    if (tid == 0 && next_k < K) {
      mbarrier_arrive_and_expect_tx_bytes(&mbar_A, bytes_tile_A);
      tma_load_tile_A(stage_tile_A(next_stage), next_k);

      mbarrier_arrive_and_expect_tx_bytes(&mbar_B, bytes_tile_B);
      tma_load_tile_B(stage_tile_B(next_stage), next_k);
    }

    // !!! IMPORTANT !!!
    // Before executing wgmma, we need to copy double_tile_A -> tile_A and double_tile_B -> tile_B
    // so that tile_A and tile_B have exact layouts as described here
    // https://docs.nvidia.com/cuda/archive/12.4.1/parallel-thread-execution/index.html#shared-memory-layout-for-wgmma-mma-async-m64nnk16
    copy_tile_A(stage_tile_A(stage), tid);
    copy_tile_B(stage_tile_B(stage), tid);

    // Ensure that all threads finish updating tile_A and tile_B
    __syncthreads();

    // Make tile_A and tile_B visible to WGMMA
    fence_proxy_async();

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

    // If next loads exist, wait for them to complete before swapping stage
    if (next_k < K) {
      mbarrier_wait(&mbar_A, /*phaseParity=*/next_stage); // NOTE: current phase LSB = 1
      mbarrier_wait(&mbar_B, /*phaseParity=*/next_stage); // NOTE: current phase LSB = 1
      stage = next_stage;
    }

    // Ensure that WGMMA finishes reading shared memory tile_A and tile_B
    // Ensure that TMA-loaded double_tile_A and double_tile_B are globally visible inside the CTA
    fence_proxy_async();
  }

  // Epilogue: write acc to tile_D
  // Map acc indices -> tile D indices
  int indices_tile_D[32];
  gen_indices(indices_tile_D, shape_tile_D, stride_tile_D, tid);

  // Copy fragments stored in acc to tile_D
  // Each thread contributes 32 fragments
  for (int i = 0; i < 32; i++) {
    int idx = indices_tile_D[i];
    tile_D[idx] = acc[i];
  }

  // Ensure that all threads finish updating tile_D
  __syncthreads();

  // Make tile_D visible to TMA engine
  fence_proxy_async();

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
// mat_B must be pre-transposed as (N, K):(K, 1)
template <typename ABType, typename DType>
void matmul_cpu(ABType* mat_A, ABType* mat_B, DType* mat_D, int stride_A[2], int stride_B[2], int stride_D[2], int M, int N, int K) {
  for (int m = 0; m < M; m++) {
    for (int n = 0; n < N; n++) {
      int idx = m * stride_D[0] + n * stride_D[1];
      for (int k = 0; k < K; k++) {
        auto mat_A_mk = static_cast<DType>(mat_A[m * stride_A[0] + k * stride_A[1]]);
        auto mat_B_kn = static_cast<DType>(mat_B[k * stride_B[0] + n * stride_B[1]]);
        mat_D[idx] += mat_A_mk * mat_B_kn;
      }
    }
  }
}


template <typename ABType, typename DType, int M_TILE, int N_TILE>
void matmul_cpu_tile(ABType* mat_A, ABType* mat_B, DType* mat_D, int stride_A[2], int stride_B[2], int stride_D[2], int M, int N, int K, int m_start, int n_start) {
  std::vector<std::thread> threads;
  for (int i = 0; i < M_TILE; i++) {
    int m = m_start + i;
    for (int j = 0; j < N_TILE; j++) {
      int n = n_start + j;
      int idx = m * stride_D[0] + n * stride_D[1];
      for (int k = 0; k < K; k++) {
        auto mat_A_mk = static_cast<DType>(mat_A[m * stride_A[0] + k * stride_A[1]]);
        auto mat_B_kn = static_cast<DType>(mat_B[k * stride_B[0] + n * stride_B[1]]);
        mat_D[idx] += mat_A_mk * mat_B_kn;
      }
    }
  }
}

constexpr unsigned int MAX_THREAD_COUNT = 1024;

template <typename ABType, typename DType, int M_TILE, int N_TILE>
void matmul_cpu_parallel(ABType* mat_A, ABType* mat_B, DType* mat_D, int stride_A[2], int stride_B[2], int stride_D[2], int M, int N, int K) {
  const int M_TILE_COUNT = M / M_TILE;
  const int N_TILE_COUNT = N / N_TILE;
  const int total_tiles = M_TILE_COUNT * N_TILE_COUNT;
  const int total_batches = total_tiles / MAX_THREAD_COUNT + ((total_tiles % MAX_THREAD_COUNT) > 0 ? 1 : 0);
  printf("Total batches of tiles to be processed: %d\n", total_batches);

  std::vector<std::thread> threads;
  int batch_count = 0;
  for (int m_tile = 0; m_tile < M_TILE_COUNT; m_tile++) {
    for (int n_tile = 0; n_tile < N_TILE_COUNT; n_tile++) {
      threads.emplace_back(
        matmul_cpu_tile<ABType, DType, M_TILE, N_TILE>, mat_A, mat_B, mat_D, stride_A, stride_B, stride_D, M, N, K, m_tile * M_TILE, n_tile * N_TILE
      );

      if (threads.size() == MAX_THREAD_COUNT) {
        for (auto& t : threads) {
          t.join();
        }
        batch_count++;
        printf("Done processing batch %d of %d tiles\n", batch_count, threads.size());
        threads.clear();
      }
    }
  }

  if (threads.size() > 0) {
    for (auto& t : threads) {
      t.join();
    }
    batch_count++;
    printf("Done processing batch %d of %d tiles\n", batch_count, threads.size());
  }
}

static std::atomic<int> g_ok(0);
static std::atomic<int> g_ng(0);

template <typename T>
void verify(T* cpu_data, T* gpu_data, int start_idx, int numElements) {
  int ok = 0;
  int ng = 0;
  constexpr T EPSILON = 1.0e-3;
  for (int i = 0; i < numElements; i++) {
    int idx = start_idx + i;
    T diff = std::abs(cpu_data[idx] - gpu_data[idx]);
    if (diff < EPSILON) {
      ok++;
    } else {
      ng++;
    }
  }

  g_ok.fetch_add(ok);
  if (ng > 0) {
    g_ng.fetch_add(ng);
  }
}


/*===---------------------------------------------------------------------------------------------------------------===*/
// main
/*===---------------------------------------------------------------------------------------------------------------===*/
int main(int argc, char** argv) {
  using DType = float;
  using ABType = bf16_t;

  int tile_count = 4;
  if (argc > 1) {
    tile_count = std::atoi(argv[1]);
  }

  if (tile_count <= 0) {
    LOG_ERROR("Tile count must be a positive integer: %d\n", tile_count);
    std::exit(1);
  }

  bool perf_mode = false;
  if (argc > 2) {
    perf_mode = std::atoi(argv[2]) == 1;
  }

  int max_round_count = 0;
  if (perf_mode) {
    max_round_count = 3;
  }

  constexpr int M_TILE = 64;
  constexpr int N_TILE = 64;
  constexpr int K_TILE = 16;

  const int M = M_TILE * tile_count;
  const int N = N_TILE * tile_count;
  const int K = K_TILE * tile_count;

  assert(tile_count <= 1024 && "Tile count must be less than 1024");

  // Init cuda
  device_init(0);
  printf("\n");

  int shape_A[] = {M, K};
  int stride_A[] = {K, 1};
  thrust::host_vector<ABType> h_A(M * K);
  thrust::device_vector<ABType> d_A(M * K);
  auto d_A_ptr = d_A.data().get();

  int shape_B[] = {K, N};
  int stride_B[] = {N, 1};
  thrust::host_vector<ABType> h_B(K * N);
  thrust::device_vector<ABType> d_B(K * N);
  auto d_B_ptr = d_B.data().get();

  int shape_D[] = {M, N};
  int stride_D[] = {N, 1};
  thrust::device_vector<DType> d_D(M * N, DType(0));
  auto d_D_ptr = d_D.data().get();

  thrust::host_vector<DType> h_D_cpu(M * N, DType(0));
  thrust::host_vector<DType> h_D_gpu(M * N, DType(0));

  // Create tensor maps
  CUtensorMap tensor_map_A{};
  bool status_A = create_tensor_map<ABType, 0>(&tensor_map_A, d_A_ptr, M, K, M_TILE, K_TILE);
  if (!status_A) {
    return EXIT_FAILURE;
  }

  CUtensorMap tensor_map_B{};
  bool status_B = create_tensor_map<ABType, 0>(&tensor_map_B, d_B_ptr, K, N, K_TILE, N_TILE);
  if (!status_B) {
    return EXIT_FAILURE;
  }

  CUtensorMap tensor_map_D{};
  bool status_D = create_tensor_map<DType, 0>(&tensor_map_D, d_D_ptr, M, N, M_TILE, N_TILE);
  if (!status_D) {
    return EXIT_FAILURE;
  }

  // Create a stream
  cudaStream_t stream;
  CUDA_CHECK_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Kernel launch config
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
  config.gridDim = dim3(gridX, gridY, 1);

    // Threadblock: 128 threads (4 warps) for one warp-group
  constexpr int THREADS_PER_BLOCK = 128;
  config.blockDim = dim3(THREADS_PER_BLOCK);
  config.stream = stream;

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> dist(0, 3);
  // std::uniform_real_distribution<float> dist(0.0, 1.0);

  FILE* file_ptr = nullptr;

  int round_ok = 0;
  int round_ng = 0;
  int round_counter = 0;
  do {
    // Reset D matrices
    std::fill(h_D_cpu.begin(), h_D_cpu.end(), DType(0));
    std::fill(h_D_gpu.begin(), h_D_gpu.end(), DType(0));

    for (int m = 0; m < M; m++) {
      for (int k = 0; k < K; k++) {
        h_A[m * stride_A[0] + k * stride_A[1]] = ABType(dist(gen));
      }
    }

    for (int k = 0; k < K; k++) {
      for (int n = 0; n < N; n++) {
        h_B[k * stride_B[0] + n * stride_B[1]] = ABType(dist(gen));
      }
    }

    if (!perf_mode) {
      file_ptr = get_file_ptr("output.txt");
      const char indices_A[] = {'m', 'k'};
      fprint_mat(file_ptr, "h_A", h_A.data(), indices_A, shape_A, stride_A);
      fprintf(file_ptr, "\n\n");

      const char indices_B[] = {'k', 'n'};
      fprint_mat(file_ptr, "h_B", h_B.data(), indices_B, shape_B, stride_B);
      fprintf(file_ptr, "\n\n");
    }

    // Transfer data from host to device
    d_A = h_A;
    d_B = h_B;

    // Launch wgmma_kernel
    CUDA_CHECK_ERROR(
      cudaLaunchKernelEx(
        &config,
        gemm_tma_wgmma_bf16_fp32<ABType, DType, M_TILE, N_TILE, K_TILE>,
        tensor_map_A, tensor_map_B, tensor_map_D,
        M, N, K
      )
    );
    printf("Launched gemm_tma_wgmma_bf16_fp32\n");

    // Copy output matrix
    cudaMemcpyAsync(h_D_gpu.data(), d_D_ptr, M * N * sizeof(DType), cudaMemcpyDeviceToHost, stream);

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    printf("Synchronize device done\n");

    // CPU
    printf("Run matmul_cpu_parallel\n");
    matmul_cpu_parallel<ABType, DType, M_TILE, N_TILE>(h_A.data(), h_B.data(), h_D_cpu.data(), stride_A, stride_B, stride_D, M, N, K);

    if (!perf_mode) {
      printf("Dumping cpu and gpu D matrices\n");
      const char indices_D[] = {'m', 'n'};
      fprint_mat(file_ptr, "h_D_cpu", h_D_cpu.data(), indices_D, shape_D, stride_D);
      fprintf(file_ptr, "\n\n");
      fprint_mat(file_ptr, "h_D_gpu", h_D_gpu.data(), indices_D, shape_D, stride_D);
    }

    // Verify
    printf("Verifying matrix D\n");
    fflush(stdout);
    // Reset global counters
    g_ok = 0; g_ng = 0;
    std::vector<std::thread> verifyThreads;
    const int kNumElements = (M * N) / MAX_THREAD_COUNT;
    for (int i = 0; i < MAX_THREAD_COUNT; i++) {
      int idx_start = i * kNumElements;
      verifyThreads.emplace_back(verify<DType>, h_D_cpu.data(), h_D_gpu.data(), idx_start, kNumElements);
    }
    
    for (auto& t : verifyThreads) {
      t.join();
    }
    printf("Matrix D: ok: %d ng: %d\n\n", g_ok.fetch_add(0), g_ng.fetch_add(0));

    round_ok += (g_ng == 0 ? 1 : 0);
    round_ng += (g_ng != 0 ? 1 : 0);
    round_counter++;
  } while(round_counter < max_round_count);

  printf("round_ok: %d round_ng: %d\n", round_ok, round_ng);
}
