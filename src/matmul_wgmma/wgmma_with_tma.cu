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

#define LOG_INFO(format_str, ...)             \
do {                                          \
  std::lock_guard<std::mutex> lk(log_mutex);  \
  printf("INFO ");                            \
  printf(format_str, ##__VA_ARGS__);          \
} while(0)

#define LOG_ERROR(format_str, ...)            \
do {                                          \
  std::lock_guard<std::mutex> lk(log_mutex);  \
  printf("ERROR ");                           \
  printf(format_str, ##__VA_ARGS__);          \
} while(0)

constexpr static int kThreadsPerWarp = 32;

/// Returns a warp-uniform value indicating the canonical warp index of the calling threads.
/// Threads within the warp must be converged.
__device__ int canonical_warp_idx_sync() {
  return __shfl_sync(0xffffffff, threadIdx.x / kThreadsPerWarp, 0);
}

// Elect one thread in the warp. 
// The elected thread gets its predicate set to true, all others obtain false.
__device__ uint32_t elect_leader_sync()
{
  uint32_t pred = 0;
  asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "elect.sync _|p, %1;\n"
      "@p mov.s32 %0, 1;\n"
    "}\n"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );

  return pred;
}

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
  // globalDim[0] is contiguous dimension
  uint64_t globalDim[] = {COLS, ROWS};
  // uint64_t globalDim[] = {ROWS, COLS, 1, 1, 1};

  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t globalStrides[] = {sizeof(DataType), COLS * sizeof(DataType)};
  // uint64_t globalStrides[] = {sizeof(DataType), ROWS * sizeof(DataType), 0, 0, 0};
  // LOG_DEBUG("globalStrides[0] = %ld\n", globalStrides[0]);

  // The boxDim is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t boxDim[] = {BOX_COLS, BOX_ROWS};
  // uint32_t boxDim[] = {BOX_ROWS, BOX_COLS, 1, 1, 1};

  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elementStrides[] = {1, 1};
  // uint32_t elementStrides[] = {1, 1, 1, 1, 1};

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
    LOG_ERROR("%s:%d: create_tensor_map does not support the provided DataType.\n", __FILE__, __LINE__);
    return false;
  }

  LOG_DEBUG("CUtensorMapDataType: %d\n", tensorDataType);

  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  if constexpr(swizzleMode == 0) {
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE;
    l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
  } else if constexpr(swizzleMode == 1) {
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B;
    l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE;
  } else if constexpr(swizzleMode == 2) {
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
    l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
  } else if constexpr(swizzleMode == 3) {
    LOG_DEBUG("swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B\n");
    swizzle = CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B;
    l2Promotion = CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
  } else {
    LOG_ERROR("%s:%d: Unsupported swizzle mode %d", __FILE__, __LINE__, swizzleMode);
    return false;
  }

  // Create the tensor descriptor.
  CUresult ret = cuTensorMapEncodeTiled(
    tensorMap,                  // CUtensorMap *tensorMap,
    tensorDataType,             // CUtensorMapDataType tensorDataType
    tensorRank,                 // cuuint32_t tensorRank,
    globalAddress,              // void *globalAddress,
    globalDim,                  // const cuuint64_t *globalDim,
    globalStrides + 1,          // const cuuint64_t *globalStrides,
    boxDim,                     // const cuuint32_t *boxDim,
    elementStrides,             // const cuuint32_t *elementStrides,

    // Interleave patterns can be used to accelerate loading of values that
    // are less than 4 bytes long.
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,

    // Swizzling can be used to avoid shared memory bank conflicts.
    swizzle,

    // L2 Promotion can be used to widen the effect of a cache-policy to a wider
    // set of L2 cache lines.
    l2Promotion,

    // Any element that is outside of bounds will be set to zero by the TMA transfer.
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE
  );

  if (ret != CUDA_SUCCESS) {
    LOG_ERROR("%s:%d: Failed to create CUtensorMap object: ROWS=%d COLS=%d BOX_ROWS=%d BOX_COLS=%d\n",
              __FILE__, __LINE__, ROWS, COLS, BOX_ROWS, BOX_COLS);
    const char* errName;
    cuGetErrorName(ret, &errName);
    const char* errMsg;
    cuGetErrorString(ret, &errMsg);
    LOG_ERROR("%s:%d: %s: %s\n", __FILE__, __LINE__, errName, errMsg);
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
__device__ uint64_t create_matrix_desc(void* smem_ptr, uint64_t stride_byte_offset) {
  uint64_t desc = 0;

  // matrix-descriptor-encode(x)
  auto mde = [](uint64_t x) -> uint64_t { return ((x & 0x3FFFF) >> 0x4); };

  uint64_t smem_addr = mde(static_cast<uint64_t>(__cvta_generic_to_shared(smem_ptr)));

  // Matrix start address: bits 13-0 (14 bits)
  desc = desc | smem_addr;

  // Leading dimension byte offset: bits 29–16 (14 bits)
  uint64_t ld_byte_offset = mde(0) << 16;
  desc = desc | ld_byte_offset;

  // Stride dimension byte offset: bits 45–32 (14 bits)
  uint64_t sd_byte_offset = mde(stride_byte_offset) << 32;
  desc = desc | sd_byte_offset;

  // Matrix base offset: bits 51–49 (3 bits)
  // This is valid for all swizzling modes except the no-swizzle mode.

  // Swizzle mode to be used: bits 63–62 (2 bits)
  uint64_t swizzle_mode = uint64_t(1) << 62;
  desc = desc | swizzle_mode;

  // 
  // Double check
  // 
#if 0
  if (blockIdx.x == 0 and threadIdx.x == 0) {
    // start_address: bits 13-0 (14 bits)
    uint64_t mask0 = 0x0000000000003FFF;
    printf("start_address: 0x%04lx\n", desc & mask0);

    // ld_byte_offset: bits 29–16 (14 bits)
    uint64_t mask1 = 0x000000003FFF0000;
    printf("ld_byte_offset: 0x%04lx\n", (desc & mask1) >> 16);

    // sd_byte_offset: bits 45–32 (14 bits)
    uint64_t mask2 = 0x00003FFF00000000;
    printf("sd_byte_offset: 0x%04lx\n", (desc & mask2) >> 32);

    // base_offset: bits 51–49 (3 bits)
    uint64_t mask3 = 0x000E000000000000;
    printf("base_offset: 0x%04lx\n", (desc & mask3) >> 49);

    // swizzle_mode: bits 63–62 (2 bits)
    uint64_t mask4 = 0xC000000000000000;
    printf("swizzle_mode: 0x%04lx\n", (desc & mask4) >> 62);
  }
#endif

  return desc;
}

// https://docs.nvidia.com/cuda/archive/12.4.0/parallel-thread-execution/index.html#matrix-fragments-for-wgmma-mma-async-m64nnk16
__device__ void gen_indices(int* indexes_tile_D, const int* shape_gmma_D, const int* stride_gmma_D, int thread_idx) {
  int baseRow = (thread_idx / 4) + (thread_idx / 32) * 8;
  int baseCol  = (thread_idx % 4) * 2;
  int k = 0;
  int idx = 0;
  for (int m = 0; m < (shape_gmma_D[0] / 32); m++) {
    int fragment_idx = k;
    for (int n = 0; n < (shape_gmma_D[1] / 8); n++) {
      idx = (baseRow + m * 8) * stride_gmma_D[0] + (baseCol + n * 8) * stride_gmma_D[1];
      indexes_tile_D[fragment_idx]   = idx;
      indexes_tile_D[fragment_idx+1] = idx + stride_gmma_D[1];
      fragment_idx += 4;
    }

    k += 2;
  }
}

// Swizzle functor
template <int BBits, int MBase, int SShift = BBits>
struct Swizzle {
  static constexpr int yyy_mask = ((1 << BBits) - 1) << (MBase + SShift);
  static constexpr int num_shft = SShift;

  __device__ int operator()(int offset) {
    int yyy_bits = (offset & yyy_mask) >> num_shft;
    return offset ^ yyy_bits; // <=> ZZZ ^ YYY
  }
};

template <typename ABType, typename DType, int M_TILE, int N_TILE, int K_TILE>
struct SharedStorage {
  // 
  // IMPORTANT NOTE: alignas(128) must precede type name
  // 
  alignas(128) ABType double_tile_A[2 * M_TILE * K_TILE];
  alignas(128) ABType double_tile_B[2 * N_TILE * K_TILE];
  alignas(128) DType tile_D[M_TILE * N_TILE];
  uint64_t mbar_A;
  uint64_t mbar_B;
  bool done[128];
};

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

  using SharedStorageType = SharedStorage<ABType, DType, M_TILE, N_TILE, K_TILE>;

  // Dynamic shared buffer
  extern __shared__ char shared_memory[];
  SharedStorageType& smem = *reinterpret_cast<SharedStorageType*>(shared_memory);

  // Double buffers double_tile_A[2 * M_TILE * K_TILE]
  constexpr uint32_t num_elems_tile_A = M_TILE * K_TILE;
  constexpr uint32_t bytes_tile_A = num_elems_tile_A * sizeof(ABType);
  ABType* double_tile_A = smem.double_tile_A;

  // Double buffers double_tile_B[2 * N_TILE * K_TILE]
  constexpr uint32_t num_elems_tile_B = N_TILE * K_TILE;
  constexpr uint32_t bytes_tile_B = num_elems_tile_B * sizeof(ABType);
  ABType* double_tile_B = smem.double_tile_B;

  // tile_D[M_TILE * N_TILE]
  DType* tile_D = smem.tile_D;

  // Two mbarriers: one for A, one for B
  // Ref: https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment
  uint64_t& mbar_A = smem.mbar_A;
  uint64_t& mbar_B = smem.mbar_B;

  // WGMMA m64n64k16
  constexpr int K_GMMA = 16;
  constexpr int K_GMMA_PARTS = K_TILE / K_GMMA;

  // Accumulators for each thread
  DType acc[32];
  #pragma unroll
  for (int i = 0; i < 32; i++) {
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

  // tile_A: M_TILE x K_TILE
  auto tma_load_tile_A = [&](ABType* stage_tile_A, int k_tile) {
    // Initiate bulk tensor copy.
    int row_A = m0;
    int col_A = k_tile;
    // printf("row_A = %d col_A = %d\n", row_A, col_A);
    cp_async_bulk_tensor_2d_global_to_shared(
      stage_tile_A,
      &tensor_map_A,
      col_A,
      row_A,
      &mbar_A
    );
  };

  auto tma_load_tile_B = [&](ABType* stage_tile_B, int k_tile) {
    int row_B = n0;
    int col_B = k_tile;
    // printf("row_B = %d col_B = %d\n", row_B, col_B);
    cp_async_bulk_tensor_2d_global_to_shared(
      stage_tile_B,
      &tensor_map_B,
      col_B,
      row_B,
      &mbar_B
    );
  };

  // // Wait for threadIdx.x - 1 to complete
  // while (threadIdx.x > 0 && not smem.done[threadIdx.x - 1]) {
  //   __nanosleep(1000);
  // }

  // Create matrix descriptors
  uint64_t desc_A[2 * K_GMMA_PARTS]; 
  uint64_t desc_B[2 * K_GMMA_PARTS];

  for (int stage = 0; stage < 2; stage++) {
    auto stage_tile_A = double_tile_A + stage * num_elems_tile_A;
    auto stage_tile_B = double_tile_B + stage * num_elems_tile_B;
    auto stage_desc_A = desc_A + stage * K_GMMA_PARTS;
    auto stage_desc_B = desc_B + stage * K_GMMA_PARTS;

    uint64_t stride_byte_offset = 8 * K_TILE * 2;
    for (int k = 0; k < K_GMMA_PARTS; k++) {
      uint64_t start_address_offset = k * K_GMMA;
      stage_desc_A[k] = create_matrix_desc<ABType>(stage_tile_A + start_address_offset, stride_byte_offset);
      stage_desc_B[k] = create_matrix_desc<ABType>(stage_tile_B + start_address_offset, stride_byte_offset);
    }
  }

  // printf("threadIdx.x = %3d init done\n", threadIdx.x);
  // smem.done[threadIdx.x] = true;

  // Thread ID
  int tid = threadIdx.x;

  // Pipeline: prefetch current_stage 0
  int current_stage = 0;
  auto stage_tile_A = double_tile_A + current_stage * num_elems_tile_A;
  auto stage_tile_B = double_tile_B + current_stage * num_elems_tile_B;

  // Initialize mbarriers once per block
  int warp_idx = threadIdx.x / kThreadsPerWarp;
  int is_leader = elect_leader_sync();

  // Do not use if (threadIdx.x == 0) because nvcc may "insert a peeling loop over all active threads, which results in warp serialization and reduced performance."
  // Ref: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html#
  if (warp_idx == 0 && is_leader) {
    // mbarrier.init.shared::cta [addr], expected_tx;
    mbarrier_init(&mbar_A, /*arrive_count=*/1); // NOTE: current phase = 0
    mbarrier_init(&mbar_B, /*arrive_count=*/1); // NOTE: current phase = 0

    // Arrive to mbarriers for first loads
    mbarrier_arrive_and_expect_tx_bytes(&mbar_A, bytes_tile_A);
    mbarrier_arrive_and_expect_tx_bytes(&mbar_B, bytes_tile_B);

    tma_load_tile_A(stage_tile_A, /*k_tile=*/0);
    tma_load_tile_B(stage_tile_B, /*k_tile=*/0);
  }

  // Syncthreads so initialized mbarriers are visible to all threads.
  __syncthreads();

  // Wait for both tiles to be ready before first compute
  // NOTE: after TMA transactions have finished, both mbar's current phases will be advanced to 1
  // The mbarrier_wait() only guarantees that the transaction count has been satisfied and 
  // the data is in shared memory somewhere — but not necessarily visible to all threads in the CTA yet.
  mbarrier_wait(&mbar_A, /*phaseParity=*/current_stage);
  mbarrier_wait(&mbar_B, /*phaseParity=*/current_stage);

  // printf("Thread %d has arrived\n", threadIdx.x);

  // This fence makes the loaded data globally visible to the CTA.
  // Only after this fence are the normal reads guaranteed 
  // to read the up-to-date data from double_tile_A/B.
  fence_proxy_async();

#if 0
  if (threadIdx.x == 0) {
    const char indices[] = {'m', 'k'};
    const int shape_tile_A[] = {M_TILE, K_TILE};
    const int stride_tile_A[] = {1, M_TILE};
    printf("\n");
    print_mat("tile_A", stage_tile_A, indices, shape_tile_A, stride_tile_A);
  }
#endif

  // Create a MMA atom
  MMA_64x64x16_F32F16F16_SS<1, 1, 1, 0, 0> mma_atom;

  // Main K loop with double buffering
  for (int k0 = 0; k0 < K; k0 += K_TILE) {
    int next_k = k0 + K_TILE;
    int next_stage = current_stage ^ 1;

    // Launch next TMA loads if there is a next slice
    if (tid == 0 && next_k < K) {
      mbarrier_arrive_and_expect_tx_bytes(&mbar_A, bytes_tile_A);
      tma_load_tile_A(double_tile_A + next_stage * num_elems_tile_A, next_k);

      mbarrier_arrive_and_expect_tx_bytes(&mbar_B, bytes_tile_B);
      tma_load_tile_B(double_tile_B + next_stage * num_elems_tile_B, next_k);
    }

    auto stage_desc_A = desc_A + current_stage * K_GMMA_PARTS;
    auto stage_desc_B = desc_B + current_stage * K_GMMA_PARTS;

    #pragma unroll
    for (int k = 0; k < K_GMMA_PARTS; k++) {
      // Enforce all threads in a warp to sync here
      canonical_warp_idx_sync();

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
        stage_desc_A[k],
        stage_desc_B[k]
      );
    }

    // Commit all wgmma ops into 1 group
    warpgroup_commit_batch();

    // Wait for MMA ops to complete
    // N=1 means don't wait if there is only 1 underway group
    warpgroup_wait<1>();

    // If next loads exist, wait for them to complete before swapping current_stage
    if (next_k < K) {
      mbarrier_wait(&mbar_A, /*phaseParity=*/next_stage); // NOTE: current phase LSB = 1
      mbarrier_wait(&mbar_B, /*phaseParity=*/next_stage); // NOTE: current phase LSB = 1
      current_stage = next_stage;
    }

    // Ensure that WGMMA finishes reading shared memory tile_A and tile_B
    // Ensure that TMA-loaded double_tile_A and double_tile_B are globally visible inside the CTA
    fence_proxy_async();
  }

  // Ensure that all MMA ops are complete
  warpgroup_wait<0>();

  // Epilogue: write acc to tile_D
  // Map acc indices -> tile D indices
  constexpr int shape_gmma_D[] = {M_TILE, N_TILE / 2};
  constexpr int stride_gmma_D[] = {N_TILE / 2, 1}; // Row-major
  int indices_gmma_D[16];
  gen_indices(indices_gmma_D, shape_gmma_D, stride_gmma_D, tid);

  // 128B swizzle mode
  auto swizzle = Swizzle<3, 4, 3>();

  // Copy fragments stored in acc to tile_D
  // Each thread contributes 32 fragments
  constexpr int num_acc = N_TILE / 4;
  for (int p = 0; p < 2; p++) {
    auto half_tile_D = tile_D + p * 64 * 32;
    for (int i = 0; i < num_acc; i++) {
      int idx = indices_gmma_D[i];
      int byte_offset = idx * sizeof(DType);
      int swizzled_byte_offset = swizzle(byte_offset);
      half_tile_D[swizzled_byte_offset / 4] = acc[p * num_acc + i];
    }
  }

  // Ensure that all threads finish updating tile_D
  __syncthreads();

  // Make tile_D visible to TMA engine
  fence_proxy_async();

  // Finally, copy tile_D back to D in global memory
  is_leader = elect_leader_sync();
  int row_D = m0;
  int col_D = n0;
  if (warp_idx == 0 && is_leader) {
    // Left half of tile_D
    cp_async_bulk_tensor_2d_shared_to_global(
      &tensor_map_D,
      col_D,
      row_D,
      tile_D
    );

    // Right half of tile_D
    cp_async_bulk_tensor_2d_shared_to_global(
      &tensor_map_D,
      col_D + 32,
      row_D,
      tile_D + 64 * 32
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

  // Ensure that all threads finish together
  __syncthreads();
}
/*===---------------------------------------------------------------------------------------------------------------===*/


// // Dmn = sum_k(Amk * Bkn) = sum_k(Amk * tBnk) = sum_k(A[m * K + k] * tB[n * K + k])
// // where, tB is transposed B
// // mat_B must be pre-transposed as (N, K):(1, N)
// template <typename ABType, typename DType>
// void matmul_cpu(ABType* mat_A, ABType* mat_B, DType* mat_D, int stride_A[2], int stride_B[2], int stride_D[2], int M, int N, int K) {
//   for (int m = 0; m < M; m++) {
//     for (int n = 0; n < N; n++) {
//       int idx = m * stride_D[0] + n * stride_D[1];
//       for (int k = 0; k < K; k++) {
//         auto mat_A_mk = static_cast<DType>(mat_A[m * stride_A[0] + k * stride_A[1]]);
//         auto mat_B_nk = static_cast<DType>(mat_B[n * stride_B[0] + k * stride_B[1]]);
//         mat_D[idx] += mat_A_mk * mat_B_nk;
//       }
//     }
//   }
// }


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
        auto mat_B_nk = static_cast<DType>(mat_B[n * stride_B[0] + k * stride_B[1]]);
        mat_D[idx] += mat_A_mk * mat_B_nk;
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
      LOG_INFO("[idx=%d] cpu != gpu: %.5f != %.5f (abs diff = %.5f)\n", idx, cpu_data[idx], gpu_data[idx], diff);
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

  int m_scale_factor = 1;
  if (argc > 1) {
    m_scale_factor = std::atoi(argv[1]);
    if (m_scale_factor <= 0) {
      LOG_ERROR("m_scale_factor must be a positive integer: %d\n", m_scale_factor);
      std::exit(1);
    }
  }

  int n_scale_factor = 1;
  if (argc > 2) {
    n_scale_factor = std::atoi(argv[2]);
    if (n_scale_factor <= 0) {
      LOG_ERROR("n_scale_factor must be a positive integer: %d\n", n_scale_factor);
      std::exit(1);
    }
  }

  int k_scale_factor = 1;
  if (argc > 3) {
    k_scale_factor = std::atoi(argv[3]);
    if (k_scale_factor <= 0) {
      LOG_ERROR("k_scale_factor must be a positive integer: %d\n", k_scale_factor);
      std::exit(1);
    }
  }

  int max_round_count = 0;
  if (argc > 4) {
    max_round_count = std::atoi(argv[4]);
    if (max_round_count <= 0) {
      LOG_ERROR("max_round_count must be a positive integer: %d\n", max_round_count);
      std::exit(1);
    }
  }

  // NOTE: Currently, only support the following tile sizes
  // M_TILE = M_GMMA (64)
  // N_TILE = N_GMMA (64)
  // K_TILE must be 64 so that 64 * sizeof(ABType) = 128 bytes
  constexpr int M_TILE = 64;
  constexpr int N_TILE = 64;
  constexpr int K_TILE = 64;

  const int M = M_TILE * m_scale_factor;
  const int N = N_TILE * n_scale_factor;
  const int K = K_TILE * k_scale_factor;

  printf("M_TILE = %d\n", M_TILE);
  printf("N_TILE = %d\n", N_TILE);
  printf("K_TILE = %d\n", K_TILE);
  printf("m_scale_factor = %d\n", m_scale_factor);
  printf("n_scale_factor = %d\n", n_scale_factor);
  printf("k_scale_factor = %d\n", k_scale_factor);
  printf("M = %d\n", M);
  printf("N = %d\n", N);
  printf("K = %d\n", K);

  bool perf_mode = max_round_count > 0;


  // Init cuda
  device_init(0);
  printf("\n");

  printf("Allocate host and device buffers\n");
  int shape_A[] = {M, K};
  int stride_A[] = {K, 1};
  thrust::host_vector<ABType> h_A(M * K);
  thrust::device_vector<ABType> d_A(M * K);
  auto d_A_ptr = d_A.data().get();

  int shape_B[] = {N, K};
  int stride_B[] = {K, 1};
  thrust::host_vector<ABType> h_B(N * K);
  thrust::device_vector<ABType> d_B(N * K);
  auto d_B_ptr = d_B.data().get();

  int shape_D[] = {M, N};
  int stride_D[] = {N, 1};
  thrust::device_vector<DType> d_D(M * N);
  auto d_D_ptr = d_D.data().get();

  thrust::host_vector<DType> h_D_cpu(M * N);
  thrust::host_vector<DType> h_D_gpu(M * N);

  // Create tensor maps
  CUtensorMap tensor_map_A{};
  bool status_A = create_tensor_map<ABType, 3>(&tensor_map_A, d_A_ptr, M, K, M_TILE, 64); // BOX_COLS = 64
  if (!status_A) {
    return EXIT_FAILURE;
  }

  CUtensorMap tensor_map_B{};
  bool status_B = create_tensor_map<ABType, 3>(&tensor_map_B, d_B_ptr, N, K, N_TILE, 64); // BOX_COLS = 64
  if (!status_B) {
    return EXIT_FAILURE;
  }

  CUtensorMap tensor_map_D{};
  bool status_D = create_tensor_map<DType, 3>(&tensor_map_D, d_D_ptr, M, N, M_TILE, N_TILE / 2);
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
  int gridX = N / N_TILE;
  int gridY = M / M_TILE;
  config.gridDim = dim3(gridX, gridY, 1);

    // Threadblock: 128 threads (4 warps) for one warp-group
  constexpr int THREADS_PER_BLOCK = 128;
  config.blockDim = dim3(THREADS_PER_BLOCK);
  config.stream = stream;

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  // std::uniform_int_distribution<int> dist(0, 3);
  std::uniform_real_distribution<float> dist(0.0, 1.0);

  FILE* file_ptr = nullptr;

  int round_ok = 0;
  int round_ng = 0;
  int round_counter = 0;
  do {
    printf("Initialize host data\n");

    // Reset D matrices
    std::fill(h_D_cpu.begin(), h_D_cpu.end(), DType(0));
    std::fill(h_D_gpu.begin(), h_D_gpu.end(), DType(0));

#if 0
    for (int m = 0; m < M; m++) {
      for (int k = 0; k < K; k++) {
        h_A[m * stride_A[0] + k * stride_A[1]] = ABType(k);
      }
    }

    for (int n = 0; n < N; n++) {
      for (int k = 0; k < K; k++) {
        h_B[n * stride_B[0] + k * stride_B[1]] = ABType(k);
      }
    }
#else
    for (int i = 0; i < M * K; i++) h_A[i] = ABType(dist(gen));
    for (int i = 0; i < N * K; i++) h_B[i] = ABType(dist(gen));
#endif

    bool dump_data = (not perf_mode) and (M * N < 1024 * 1024);

    if (dump_data) {
      file_ptr = get_file_ptr("output.txt");
      const char indices_A[] = {'m', 'k'};
      fprint_mat(file_ptr, "h_A", h_A, indices_A, shape_A, stride_A);
      fprintf(file_ptr, "\n\n");

      const char indices_B[] = {'n', 'k'};
      fprint_mat(file_ptr, "h_B", h_B, indices_B, shape_B, stride_B);
      fprintf(file_ptr, "\n\n");
    }

    // Transfer data from host to device
    printf("Transfer data from host to device\n");
    d_A = h_A;
    d_B = h_B;

    // Reset
    d_D = h_D_cpu;

    // Dynamic shared memory
    std::size_t dyn_smem_bytes = 2 * M_TILE * K_TILE * sizeof(ABType) + 
                                 2 * N_TILE * K_TILE * sizeof(ABType) +
                                 M_TILE * N_TILE * sizeof(DType) +
                                 sizeof(uint64_t) + sizeof(uint64_t) + 128;
    config.dynamicSmemBytes = dyn_smem_bytes;
    
    auto kernel_ptr = gemm_tma_wgmma_bf16_fp32<ABType, DType, M_TILE, N_TILE, K_TILE>;
    CUDA_CHECK_ERROR(
      cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dyn_smem_bytes
      )
    );

    // Launch wgmma_kernel
    CUDA_CHECK_ERROR(
      cudaLaunchKernelEx(
        &config,
        kernel_ptr,
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

    if (dump_data) {
      printf("Dumping cpu and gpu D matrices\n");
      const char indices_D[] = {'m', 'n'};
      fprint_mat(file_ptr, "h_D_cpu", h_D_cpu, indices_D, shape_D, stride_D);
      fprintf(file_ptr, "\n\n");
      fprint_mat(file_ptr, "h_D_gpu", h_D_gpu, indices_D, shape_D, stride_D);
    }

    // Verify
    printf("Verifying matrix D\n");
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
