#include <cstdio>
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

using bf16_t = __nv_bfloat16;
using half_t = __nv_half;

static std::mutex log_mutex;
#if defined(DEBUG_MODE)
#define LOG_DEBUG(format_str, ...)            \
do {                                          \
  std::lock_guard<std::mutex> lk(log_mutex);  \
  printf("DEBUG_MODE ");                           \
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

__device__ inline void sync_threads_256() 
{
  asm volatile(
    "\n"
    "\tbar.sync\t0, 256;"
    "\n"
  );
}

__device__ inline void sync_threads_64() 
{
  asm volatile(
    "\n"
    "\tbar.sync\t2, 64;"
    "\n"
  );
}

__device__ inline void sync_warp() 
{
  asm volatile(
    "\n"
    "\tbar.warp.sync\t-1;"
    "\n"
  );
}

__device__ inline void sync_cta_unaligned() 
{
  asm volatile(
    "\n"
    "\tbarrier.sync\t1;"
    "\n"
  );
}

/// Returns a warp-uniform value indicating the canonical warp index of the calling threads.
/// Threads within the warp must be converged.
__device__ inline int canonical_warp_idx_sync() {
  return __shfl_sync(0xffffffff, threadIdx.x / kThreadsPerWarp, 0);
}

// Elect one thread in the warp. 
// The elected thread gets its predicate set to true, all others obtain false.
__device__ inline uint32_t elect_leader_sync()
{
  uint32_t pred = 0;
  asm volatile(
    "\n"
    "\t{\n"
      "\t\t.reg .pred p;\n"
      "\t\telect.sync _|p, %1;\n"
      "\t\t@p mov.s32 %0, 1;\n"
    "\t}\n"
    : "+r"(pred)
    : "r"(0xFFFFFFFF)
  );

  return pred;
}

// Allocate TMEM
__device__ inline void tcgen05_alloc_tmem_cols_64(uint32_t pred, uint32_t* tmem_handle_ptr)
{
  asm volatile(
    "\n"
    "\t{\n"
    "\t\t.reg .pred %p;\n"
    "\t\tsetp.eq.b32\t%p, %0, 1;\n"
    "\t\t@%p tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%1], 64;\n"
    "\t}\n"
    : 
    : "r"(pred), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(tmem_handle_ptr)))
    : "memory"
  );
}

__device__ inline void tcgen05_relinquish_alloc_permit(uint32_t pred)
{
  asm volatile(
    "\n"
    "\t{\n"
    "\t\t.reg .pred %p;\n"
    "\t\tsetp.eq.b32\t%p, %0, 1;\n"
    "\t\t@%p tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;\n"
    "\t}\n"
    :
    : "r"(pred)
    : "memory"
  );
}

__device__ inline void tcgen05_reset_tmem(uint32_t tmem_handle)
{
  asm volatile(
    "\n"
    "\ttcgen05.st.sync.aligned.16x32bx2.x16.b32 [%0 + 0], 16, {%1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1, %1};"
    "\n"
    :
    : "r"(tmem_handle), "r"(0)
    : "memory"
  );
}

__device__ inline void tcgen05_load_tmem(float* acc, uint32_t tmem_handle)
{
  asm volatile(
    "\n"
    "\ttcgen05.ld.sync.aligned.16x32bx2.x16.b32 "
    "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16 + 0], 16;"
    "\n"
    : "=f"(acc[0]), "=f"(acc[1]), "=f"(acc[2]),  "=f"(acc[3]),  "=f"(acc[4]),  "=f"(acc[5]),  "=f"(acc[6]),  "=f"(acc[7]),
      "=f"(acc[8]), "=f"(acc[9]), "=f"(acc[10]), "=f"(acc[11]), "=f"(acc[12]), "=f"(acc[13]), "=f"(acc[14]), "=f"(acc[15])
    : "r"(tmem_handle)
    : "memory"
  );
}

__device__ inline void tcgen05_dealloc_tmem(uint32_t pred, uint32_t tmem_handle)
{
  asm volatile(
    "\n"
    "\t{\n"
    "\t\t.reg .pred %p;\n"
    "\t\tsetp.eq.b32\t%p, %0, 1;\n"
    "\t\t@%p tcgen05.dealloc.cta_group::1.sync.aligned.b32 %1, 64;\n"
    "\t}\n"
    :
    : "r"(pred), "r"(tmem_handle)
    : "memory"
  );
}

// MMAv5
// [ %2 + 0 ] is primarily for canonical consistency and architectural alignment.
__device__ inline void tcgen05_mma_m64n64k16f16(
  uint32_t pred, 
  uint32_t tmem_handle, 
  uint64_t a_desc, 
  uint64_t b_desc, 
  uint32_t inst_desc, 
  uint32_t acc_pred
)
{
  asm volatile(
    "\n"
    "\t{\n"
    "\t\t.reg .pred %%p1;\n"
    "\t\t.reg .pred %%p2;\n"
    "\t\tsetp.eq.u32\t%%p1, %0, 1;\n"
    "\t\tsetp.eq.u32\t%%p2, %1, 1;\n"
    "\t\t@%%p1 tcgen05.mma.cta_group::1.kind::f16 [ %2 + 0 ], %3, %4, %5, %%p2;\n"
    "\t}\n"
    :
    : "r"(pred),
      "r"(acc_pred),
      "r"(tmem_handle), 
      "l"(a_desc),
      "l"(b_desc),
      "r"(inst_desc)
    : "memory"
  );
}

// Marks the commit point for one or more sized batch of warpgroup MMAs.
__device__ inline void tcgen05_commit_batch(uint32_t pred, const void* mbarrier_ptr)
{
  asm volatile(
    "\n"
    "\t{\n"
    "\t\t.reg .pred %p;\n"
    "\t\tsetp.eq.b32\t%p, %0, 1;\n"
    "\t\t@%p tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%1];\n"
    "\t}\n"
    :
    : "r"(pred), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr)))
    : "memory"
  );
}

__device__ inline void tcgen05_wait_store()
{
  asm volatile("\n\ttcgen05.wait::st.sync.aligned;\n" : : : "memory");
}

__device__ inline void tcgen05_wait_load()
{
  asm volatile("\n\ttcgen05.wait::ld.sync.aligned;\n" : : : "memory");
}

__device__ inline void fence_proxy_async() {
  asm volatile("\n\tfence.proxy.async.shared::cta;\n" : : : "memory");
}

__device__ inline void mbarrier_init(uint32_t first_tid_pred, const void* mbarrier_ptr, uint32_t arrive_count)
{
  asm volatile(
    "\n"
    "\t{\n"
    "\t\t.reg .pred %p;\n"
    "\t\tsetp.eq.b32\t%p, %0, 1;\n"
    "\t\t@%p mbarrier.init.shared::cta.b64 [%1], %2;\n"
    "\t}\n"
    :
    : "r"(first_tid_pred), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr))), "r"(arrive_count)
    : "memory"
  );
}

__device__ inline void mbarrier_arrive(uint32_t first_tid_pred, const void* mbarrier_ptr) {
  asm volatile(
    "\n"
    "\t{\n"
    "\t\t.reg .pred %p;\n"
    "\t\tsetp.eq.b32\t%p, %0, 1;\n"
    "\t\t@%p mbarrier.arrive.shared::cta.b64 _, [%1];\n"
    "\t}\n"
    :
    : "r"(first_tid_pred), "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr)))
    : "memory"
  );
}

// Performs an arrive operation + expected transaction count
__device__ inline void mbarrier_arrive_and_expect_tx_bytes(uint32_t first_tid_pred, void const* mbarrier_ptr, uint32_t bytes) {
  asm volatile(
    "\n"
    "\t{\n"
    "\t\t.reg .pred %p;\n"
    "\t\tsetp.eq.b32\t%p, %0, 1;\n"
    "\t\t@%p mbarrier.arrive.expect_tx.shared.b64 _, [%1], %2;\n"
    "\t}\n"
    :
    : "r"(first_tid_pred),
      "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr))),
      "r"(bytes)
    : "memory"
  );
}

__device__ inline void mbarrier_wait(const void* mbarrier_ptr, uint32_t phaseParity) {
  // Arbitrarily large timer value after which try-wait expires and re-tries.
  uint32_t ticks = 0x989680;
  asm volatile(
    "\n"
    "\t{\n"
      "\t\t.reg .pred complete;\n"
      "\t\tLAB_WAIT:\n"
      "\t\t\tmbarrier.try_wait.parity.shared.b64 complete, [%0], %1, %2;\n"
      "\t\t\t@!complete bra.uni LAB_WAIT;\n"
    "\t}\n"
    :
    : "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr))), "r"(phaseParity), "r"(ticks)
  );
}

__device__ inline void cp_async_bulk_tensor_2d_global_to_shared(uint32_t pred, const void* dst_ptr, const void* tensorMap, int col, int row, const void* mbarrier_ptr) {
    asm volatile(
      "\n"
      "\t{\n"
      "\t\t.reg .pred %p;\n"
      "\t\tsetp.eq.b32\t%p, %0, 1;\n"
      "\t\t@%p cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%1], [%2, {%3, %4}], [%5];\n"
      "\t}\n"
         :
         : "r"(pred), 
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_ptr)))
           "l"(reinterpret_cast<uint64_t>(tensorMap)),
           "r"(col),
           "r"(row),
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr)))
         : "memory"
    );
}

__device__ inline void cp_async_bulk_tensor_2d_shared_to_global(uint32_t pred, const void* tensorMap, int col, int row, const void* mbarrier_ptr) {
    asm volatile(
      "\n"
      "\t{\n"
      "\t\t.reg .pred %p;\n"
      "\t\tsetp.eq.b32\t%p, %0, 1;\n"
      "\t\t@%p cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%1, {%2, %3}], [%4];\n"
      "\t}\n"
         :
         : "r"(pred),
           "l"(reinterpret_cast<uint64_t>(tensorMap)),
           "r"(col), 
           "r"(row),
           "r"(static_cast<uint32_t>(__cvta_generic_to_shared(mbarrier_ptr)))
         : "memory"
    );
}

__device__ inline void cp_async_bulk_commit_group() {
  asm volatile("\n\tcp.async.bulk.commit_group;\n" : : : "memory");
}

template <int N>
__device__ inline void cp_async_bulk_wait_group_read() {
  asm volatile("\n\tcp.async.bulk.wait_group.read %0;\n" : : "n"(N) : "memory");
}


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
  } else if constexpr(std::is_same<DataType, half_t>()) {
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
__device__ uint64_t create_matrix_desc(void* smem_ptr, uint64_t sbo) {
  uint64_t desc = 0;

  // matrix-descriptor-encode(x)
  auto mde = [](uint64_t x) -> uint64_t { return ((x & 0x3FFFF) >> 0x4); };

  uint64_t smem_addr = mde(static_cast<uint64_t>(__cvta_generic_to_shared(smem_ptr)));

  // Matrix start address: bits 13-0 (14 bits)
  desc = desc | smem_addr;

  // Leading dimension byte offset: bits 29–16 (14 bits)
  // This bit field is ignored when a swizzling mode is set,
  // so we set it to 0
  uint64_t ld_byte_offset = mde(0) << 16;
  desc = desc | ld_byte_offset;

  // Stride dimension byte offset: bits 45–32 (14 bits)
  uint64_t sd_byte_offset = mde(sbo) << 32;
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

/* 
For each warp, thread mapping to TMEM is as follows:

row/col 0                      15                      31
        -------------------------------------------------
      0 |         tid=0         |         tid=16        |
        -------------------------------------------------
      1 |         tid=1         |         tid=17        |
        -------------------------------------------------
      2 |         tid=2         |         tid=18        |
        -------------------------------------------------
        .                                               .
        .                                               .
        .                                               .
        -------------------------------------------------
     15 |         tid=15        |         tid=31        |
        -------------------------------------------------

Each thread holds 16 elements of matrix D.
Based on this mapping, we generate linear indices for matrix D as follows:
*/
__device__ void gen_indices(int* d_mma_indices, const int* d_mma_shape, const int* d_mma_stride, int tid) {
  int warp_id = tid / kThreadsPerWarp;
  int p = (tid - warp_id * kThreadsPerWarp) < 16 ? 0 : 1;
  
  // Each warp fills 16 rows
  int base_row = (warp_id % 4) * 16;

  // Left half-warp is responsible for 16x16 tile
  // Right half-warp is responsible for next 16x16 tile
  int base_col = p * 16;

  int row_offset = tid % 16;
  
  // Each thread is responsible for 1 row and 16 cols
  for (int i = 0; i < 16; i++) {
    int idx = (base_row + row_offset) * d_mma_stride[0] + (base_col + i) * d_mma_stride[1];
    d_mma_indices[i] = idx;
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

template <typename ABType, typename DType, int M_TILE, int N_TILE, int K_TILE, int NUM_STAGES>
struct SharedStorage {
  // 
  // IMPORTANT NOTE: alignas(128) must precede type name
  // 
  alignas(128) ABType a_stage_tiles[NUM_STAGES * M_TILE * K_TILE];
  alignas(128) ABType b_stage_tiles[NUM_STAGES * N_TILE * K_TILE];
  alignas(128) DType d_tile[M_TILE * N_TILE];
  uint64_t consumer_mbars[NUM_STAGES];
  uint64_t producer_mbars[NUM_STAGES];
  uint64_t final_mbar;
  uint32_t tmem_handle;
#if defined(DEBUG_MODE)
  bool done[256];
#endif
};

/*===---------------------------------------------------------------------------------------------------------------===*/
// tma_mmav5_fn
/*===---------------------------------------------------------------------------------------------------------------===*/
// Kernel uses TMA mbarriers (64B) and smem
template <typename ABType, typename DType, int M_TILE, int N_TILE, int K_TILE, int NUM_STAGES>
__device__ void tma_mmav5_fn(
  const CUtensorMap& a_tensor_map,
  const CUtensorMap& b_tensor_map,
  const CUtensorMap& d_tensor_map,
  int M, int N, int K
) {

#if defined(__CUDA_ARCH__)
  #pragma message("__CUDA_ARCH__: " TO_STR(__CUDA_ARCH__))
  static_assert(__CUDA_ARCH__ >= 1000, "Requires __CUDA_ARCH__ >= 100");
#endif

  static_assert(std::is_same<ABType, half_t>() || std::is_same<ABType, bf16_t>(), "ABType is not supported!");

  using SharedStorageType = SharedStorage<ABType, DType, M_TILE, N_TILE, K_TILE, NUM_STAGES>;

  // Dynamic shared buffer
  extern __shared__ char global_smem[];
  SharedStorageType& smem = *reinterpret_cast<SharedStorageType*>(global_smem);

  // Stage buffer a_stage_tiles[NUM_STAGES * M_TILE * K_TILE]
  constexpr uint32_t a_tile_num_elems = M_TILE * K_TILE;
  constexpr uint32_t a_tile_num_bytes = a_tile_num_elems * sizeof(ABType);
  ABType* a_stage_tiles = smem.a_stage_tiles;

  // Stage buffer b_stage_tiles[NUM_STAGES * N_TILE * K_TILE]
  constexpr uint32_t b_tile_num_elems = N_TILE * K_TILE;
  constexpr uint32_t b_tile_num_bytes = b_tile_num_elems * sizeof(ABType);
  ABType* b_stage_tiles = smem.b_stage_tiles;

  // d_tile[M_TILE * N_TILE]
  DType* d_tile = smem.d_tile;

  // A pointer to TMEM handle
  uint32_t* tmem_handle_ptr = &smem.tmem_handle;

  // Reset TMEM handle to zero
  *tmem_handle_ptr = 0;

  // Consumer and producer mbarriers
  // Ref: https://docs.nvidia.com/cuda/parallel-thread-execution/#parallel-synchronization-and-communication-instructions-mbarrier-size-alignment
  uint64_t* consumer_mbars = smem.consumer_mbars;
  uint64_t* producer_mbars = smem.producer_mbars;

  // Final mbarrier
  uint64_t* final_mbar = &smem.final_mbar;

  // MMA m64n64k16
  constexpr int K_MMA_SIZE = 16;

  static_assert(K_TILE % K_MMA_SIZE == 0, "K_TILE should be multiple of 16");
  constexpr int K_MMA_PARTS = K_TILE / K_MMA_SIZE;

  // Accumulators for each thread
  DType acc[16];
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    acc[i] = DType(0);
  }

  // Compute block tile indices
  int block_m = blockIdx.y; // y is row
  int block_n = blockIdx.x; // x is col
  int m0 = block_m * M_TILE;
  int n0 = block_n * N_TILE;

  // Bounds check for full tiles (simple version)
  if (K <= 0 || (m0 + M_TILE) > M || (n0 + N_TILE) > N) {
    return;
  }

  // Compute k_tiles
  int k_tiles = (K + K_TILE - 1) / K_TILE;

  // Thread ID
  int tid = threadIdx.x;
  int warp_id = canonical_warp_idx_sync();

  // Sync all threads
  __syncthreads();


  //====================================================================================
  // Warp 0-7
  //====================================================================================
  if (warp_id < 8) {
    // 
    // 1. Allocate a TMEM tile of 128 rows by 64 columns (128x64)
    // 

    // First tid predicate
    uint32_t first_tid_pred = (tid == 0) ? 1 : 0;

    // First wrap predicate
    uint32_t first_warp_pred = (warp_id == 0) ? 1 : 0;

    // Sync 256 threads (warp 0-7)
    sync_threads_256();

    // Allocate TMEM
    tcgen05_alloc_tmem_cols_64(first_warp_pred, tmem_handle_ptr);

    // Sync 256 threads (warp 0-7)
    sync_threads_256();

    // Release TMEM allocation permit
    tcgen05_relinquish_alloc_permit(first_warp_pred);


    // 
    // 2. Reset TMEM tile
    //

    uint32_t is_leader = elect_leader_sync();

    // Load TMEM address
    uint32_t base_tmem_handle = *tmem_handle_ptr;

    // Compute row offset
    uint32_t row_offset = (warp_id << 21) & 6291456; // 6291456 = 2^21 + 2^22;

    // Compute col offset
    uint32_t col_offset = (warp_id * 8) & 32;

    // Compute TMEM address
    uint32_t tmem_handle = base_tmem_handle + row_offset + col_offset;

    // Ensure that tmem_handle is identical accross 256 threads
    sync_threads_256();

    // Reset TMEM
    tcgen05_reset_tmem(tmem_handle);
    tcgen05_wait_store();


    // 
    // 3. Init Consumer mbarriers
    // 4. Init Producer mbarriers
    // 

    // Init Consumer mbarriers
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(first_tid_pred, consumer_mbars + i, 1);
    }

    // Init Producer mbarriers
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_init(first_tid_pred, producer_mbars + i, 1);
    }


    // 
    // 5. Prime Consumer mbarriers
    // 
    for (int i = 0; i < NUM_STAGES; i++) {
      mbarrier_arrive(first_tid_pred, consumer_mbars + i);
    }

    // Init the final mbarrier
    mbarrier_init(first_tid_pred, final_mbar, 1);


    // 
    // 6. Store branch index value `33554689` to `[global_smem+82152]`
    // 
    // *warp_branch_indices = 33554689;

  
    // 
    // 7. Arrive at 1st CTA sync point
    // 
    sync_cta_unaligned();

    
    // 
    // 8. Arrive at 2nd CTA sync point
    // 
    sync_cta_unaligned();

    
    // 
    // 9. Arrive at 3rd CTA sync point
    // 
    sync_cta_unaligned();


    // 
    // 10. Wait for the final mbarrier to be satisfied
    // 

    // Warp 0-7 wait for the final mbarrier to be satisfied.
    // Once it is satisfied, MMA ops have consumed all k_tiles along K dimension
    mbarrier_wait(final_mbar, 0);


    // 
    // 11. Load matrix D from TMEM to registers
    // 
    // Sync warp 0-7
    sync_threads_256();

    tcgen05_load_tmem(acc, tmem_handle);
    tcgen05_wait_load();

    // Sync warp 0-7
    sync_threads_256();

  #if defined(DEBUG_MODE)
    // Wait for threadIdx.x - 1 to complete
    done = false;
    while (not first_tid_pred && not done) {
      __nanosleep(1000);
      done = ((volatile bool*)smem.done)[tid - 1];
    }

    printf("[tid = %d] acc: ", tid);
    #pragma unroll(1)
    for (int i = 0; i < 16; i++) {
      printf("%10.3f", acc[i]);
    }
    printf("\n");
    smem.done[tid] = true;
  #endif

    // 
    // 12. Store registers into SMEM in 128B swizzling mode
    // 

    // Map acc indices -> tile D indices
    constexpr int d_mma_shape[] = {M_TILE, N_TILE / 2};
    constexpr int d_mma_stride[] = {N_TILE / 2, 1}; // Row-major
    int d_mma_indices[16];
    gen_indices(d_mma_indices, d_mma_shape, d_mma_stride, tid);

    // 128B swizzle mode
    auto swizzle = Swizzle<3, 4, 3>();

    // Copy fragments stored in acc to d_tile
    // Each thread contributes 16 fragments
    int p = warp_id < 4 ? 0 : 1;
    auto half_tile = d_tile + p * M_TILE * (N_TILE / 2);

    #pragma unroll
    for (int i = 0; i < 16; i++) {
      int idx = d_mma_indices[i];
      int byte_offset = idx * sizeof(DType);
      int swizzled_byte_offset = swizzle(byte_offset);
      half_tile[swizzled_byte_offset / sizeof(DType)] = acc[i];
    }

    // Ensure that all 256 threads finish updating d_tile together
    sync_threads_256();

  #if 0
    if (first_tid_pred) {
      printf("\nd_tile:\n");
      for (int m = 0; m < M_TILE; m++) {
        printf("m = %3d:", m);
        for (int n = 0; n < (N_TILE / 2); n++) {
          printf("%12.3f", d_tile[m * (N_TILE / 2) + n]);
        }
        printf("\n");
      }
      printf("\n\n");
    }
  #endif

    // Make d_tile visible to TMA engine
    fence_proxy_async();


    // 
    // 13. TMA matrix D from SMEM to GMEM
    // 

    // uint32_t is_leader = elect_leader_sync();
    int d_row = block_m * M_TILE;
    int d_col = block_n * N_TILE;
    uint32_t tma_pred0 = (warp_id == 0 && is_leader) ? 1 : 0;
    uint32_t tma_pred1 = (warp_id == 1 && is_leader) ? 1 : 0;

    // Left half of d_tile
    cp_async_bulk_tensor_2d_shared_to_global(
      tma_pred0,
      &d_tensor_map,
      d_col,
      d_row,
      d_tile
    );

    // Right half of d_tile
    cp_async_bulk_tensor_2d_shared_to_global(
      tma_pred1,
      &d_tensor_map,
      d_col + (N_TILE / 2),
      d_row,
      d_tile + M_TILE * (N_TILE / 2)
    );

    // Sync 256 threads
    sync_threads_256();

    // Ref1: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#completion-mechanisms-for-asynchronous-copy-operations
    // Ref2: https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async-bulk-commit-group
    // Create a "bulk async-group" out of the previous bulk copy operation.
    // Purpose: for tracking the completion of this group
    cp_async_bulk_commit_group();

    // Wait for all the asynchronous operations belonging to the bulk async-group are complete.
    // `0` means the executing thread waits on all the prior bulk async-groups to complete.
    cp_async_bulk_wait_group_read<0>();


    // 
    // 14. Deallocate TMEM
    // 
    uint32_t tmem_dealloc_pred = (warp_id == 0) ? 1 : 0;
    tcgen05_dealloc_tmem(tmem_dealloc_pred, tmem_handle);


    // 
    // 15. Update warp_branch_indices to `50529027`
    // 
    // *warp_branch_indices = 50529027;


    // 
    // 16. Arrive at the 4th CTA sync point
    // 
    sync_cta_unaligned();
  } 


  //====================================================================================
  // Warp 8-9
  //====================================================================================
  else if (warp_id == 8 || warp_id == 9) {
    // 
    // Prologue
    // 

    // 1. Arrive at the 1st CTA sync point
    sync_cta_unaligned();

    // 2. Arrive at the 2nd CTA sync point
    sync_cta_unaligned();

    // TMA coordinates for A
    int a_col = 0;
    int a_row = block_m * M_TILE;

    // TMA coordinates for B
    int b_col = 0;
    int b_row = block_n * N_TILE;

    // Parity phase
    int parity_phase = 0;

    // Stage counter
    int stage_counter = 0;

    // Compute the first tid of warp 8 predicate
    uint32_t first_tid_pred = (tid == 256) ? 1 : 0;

    // Init k_tile_counter
    int k_tile_counter = k_tiles;

    // 
    // Enter Copy Loop
    // 
    #pragma unroll(1)
    while (k_tile_counter > 0) {
      // Ensure that stage_counter and parity_phase are identical
      // accross 64 threads of warp 8-9
      sync_threads_64();

      // 1. Wait for Consumer mbarriers to be satisfied
      mbarrier_wait(consumer_mbars + stage_counter, parity_phase);

      // 2. TMA 64x64xf16 tile_m and tile_n to global_smem

      // Expect 2*64*64*2 bytes (2 tiles of 64x64xf16 for A and B)
      mbarrier_arrive_and_expect_tx_bytes(
        first_tid_pred, 
        producer_mbars + stage_counter,
        a_tile_num_bytes + b_tile_num_bytes
      );

      // Copy tile_m
      cp_async_bulk_tensor_2d_global_to_shared(
        first_tid_pred,
        a_stage_tiles + stage_counter * a_tile_num_elems,
        &a_tensor_map,
        a_col,
        a_row,
        producer_mbars + stage_counter
      );

      // Copy tile_n
      cp_async_bulk_tensor_2d_global_to_shared(
        first_tid_pred,
        b_stage_tiles + stage_counter * a_tile_num_elems,
        &b_tensor_map,
        b_col,
        b_row,
        producer_mbars + stage_counter
      );

      // Is this the last stage?
      bool is_last_stage = (stage_counter + 1 == NUM_STAGES);

      // Update parity phase
      parity_phase = is_last_stage ? ~parity_phase : parity_phase;

      // Update stage_counter
      stage_counter = is_last_stage ? 0 : stage_counter + 1;

      // Update k_tile_counter
      k_tile_counter -= 1;

      // Update column indices for TMA
      a_col += K_TILE;
      b_col += K_TILE;
    }

    // 
    // Epilogue
    // 

    // 1. Arrive at the 3rd CTA sync point
    sync_cta_unaligned();
  }


  //====================================================================================
  // Warp 10
  //====================================================================================
  else if (warp_id == 10) {
    // 
    // Prologue
    // 

    // 1. Arrive at the 1st CTA sync point
    sync_cta_unaligned();

    // 2. Arrive at the 2nd CTA sync point
    sync_cta_unaligned();

    // Init variables
    int stage_counter = 0;
    int parity_phase = 0;
    int k_tile_counter = k_tiles - 1;
    uint32_t tmem_handle = *tmem_handle_ptr;

    // Sync warp 10
    sync_warp();

    // 3. Wait for the first Producer mbarrier to be satisfied
    mbarrier_wait(producer_mbars + stage_counter, parity_phase);

    // 4. MMA the first 64x64xf16 tile_m and tile_n
    auto a_stage_tile = a_stage_tiles + stage_counter * a_tile_num_elems;
    auto b_stage_tile = b_stage_tiles + stage_counter * b_tile_num_elems;

    // Each core matrix has 8 rows
    // Strided by K_TILE along M or N dimension
    // Each bf16/fp16 element occupies 2 bytes
    constexpr uint64_t STRIDE_BYTE_OFFSET = 8 * K_TILE * sizeof(ABType);

    // 68157456 = 000|00100|0|001000|00000000000010000
    // bits 22-17 = 001000 => N = 2^3 *  8 = 64
    // bits 28-24 =  00100 => M = 2^2 * 16 = 64
    uint32_t INST_DESC = 0;
    if constexpr(std::is_same<ABType, half_t>()) {
      INST_DESC = 68157456;
    } else if constexpr(std::is_same<ABType, bf16_t>()) {
      // Set atype and btype
      INST_DESC = 68157456 | ((1 << 7) + (1 << 10));
    } else {
      // Unreachable
    }

    // 32 threads in warp 10 is also synced at this point
    uint32_t is_leader = elect_leader_sync();

  #if 0
    if (is_leader) {
      printf("\nk_tile_counter = %d\n", k_tile_counter);
      printf("\na_stage_tile:\n");
      for (int m = 0; m < M_TILE; m++) {
        printf("m = %3d:", m);
        for (int k = 0; k < K_TILE; k++) {
          if constexpr(std::is_same<ABType, half_t>()) {
            printf("%6.1f", __half2float(a_stage_tile[m * K_TILE + k]));
          } else {
            printf("%6.1f", __bfloat162float(a_stage_tile[m * K_TILE + k]));
          }
        }
        printf("\n");
      }
      printf("\n\n");

      printf("\nb_stage_tile:\n");
      for (int n = 0; n < N_TILE; n++) {
        printf("n = %3d:", n);
        for (int k = 0; k < K_TILE; k++) {
          if constexpr(std::is_same<ABType, half_t>()) {
            printf("%6.1f", __half2float(b_stage_tile[n * K_TILE + k]));
          } else {
            printf("%6.1f", __bfloat162float(b_stage_tile[n * K_TILE + k]));
          }
        }
        printf("\n");
      }
      printf("\n\n");
    }
  #endif

    #pragma unroll
    for (int k_part = 0; k_part < K_MMA_PARTS; k_part++) {
      uint64_t start_address_offset = k_part * K_MMA_SIZE;
      uint64_t a_desc = create_matrix_desc<ABType>(a_stage_tile + start_address_offset, STRIDE_BYTE_OFFSET);
      uint64_t b_desc = create_matrix_desc<ABType>(b_stage_tile + start_address_offset, STRIDE_BYTE_OFFSET);
      uint32_t acc_pred = (k_part == 0) ? 0 : 1;

      // Only the leader thread executes the PTX instruction
      tcgen05_mma_m64n64k16f16(is_leader, tmem_handle, a_desc, b_desc, INST_DESC, acc_pred);
    }

    // 5. Satisfy the first Consumer mbarrier
    // which signifies that the first 64x64xf16 tile_m and tile_n 
    // is already consumed
    tcgen05_commit_batch(is_leader, consumer_mbars + stage_counter);

    // 6. Satisfy the final mbarrier if k_tile_counter == 0
    uint32_t final_mbar_pred = (k_tile_counter == 0 && is_leader) ? 1 : 0;
    tcgen05_commit_batch(final_mbar_pred, final_mbar);

    // Update stage_counter
    stage_counter += 1;

    // 
    // Enter MMA Loop
    // 
    #pragma unroll(1)
    while (k_tile_counter > 0) {
      // Sync warp 10 to ensure that stage_counter and parity_phase are identical
      // accross 32 threads
      sync_warp();

      a_stage_tile = a_stage_tiles + stage_counter * a_tile_num_elems;
      b_stage_tile = b_stage_tiles + stage_counter * b_tile_num_elems;

      // 1. Wait for Producer mbarriers to be satisfied.
      mbarrier_wait(producer_mbars + stage_counter, parity_phase);

      // 2. MMA tile_m and tile_n (64x64xf16)
      #pragma unroll
      for (int k_part = 0; k_part < K_MMA_PARTS; k_part++) {
        uint64_t start_address_offset = k_part * K_MMA_SIZE;
        uint64_t a_desc = create_matrix_desc<ABType>(a_stage_tile + start_address_offset, STRIDE_BYTE_OFFSET);
        uint64_t b_desc = create_matrix_desc<ABType>(b_stage_tile + start_address_offset, STRIDE_BYTE_OFFSET);

        // Only the leader thread executes the PTX instruction
        tcgen05_mma_m64n64k16f16(is_leader, tmem_handle, a_desc, b_desc, INST_DESC, 1);
      }

      // 3. Satisfy Consumer mbarriers
      tcgen05_commit_batch(is_leader, consumer_mbars + stage_counter);

      // Update final_mbar_pred
      final_mbar_pred = (k_tile_counter == 1 && is_leader) ? 1 : 0;

      // Satisfy the final mbarrier if k_tile_counter == 1 (the final (tile_m, tile_n) was consumed)
      tcgen05_commit_batch(final_mbar_pred, final_mbar);

      // Is this the last stage?
      bool is_last_stage = (stage_counter + 1 == NUM_STAGES);

      // Flip parity_phase if this is the last stage
      // else keep it as it is
      parity_phase = is_last_stage ? ~parity_phase : parity_phase;

      // Reset stage_counter to 0 if this is the last stage
      // else increment stage_counter by 1
      stage_counter = is_last_stage ? 0 : stage_counter + 1;

      // Decrement k_tile_counter by 1
      k_tile_counter -= 1;
    }


    // 
    // Epilogue
    // 

    // 1. Arrive the 3rd CTA sync point
    sync_cta_unaligned();
  } 


  //====================================================================================
  // Warp 11
  //====================================================================================
  else {
    // 1. Arrive at 1st CTA sync point
    sync_cta_unaligned();

    // 2. Arrive at 2nd CTA sync point
    sync_cta_unaligned();

    // 3. Arrive at 3rd CTA sync point
    sync_cta_unaligned();
  }

  // Warp 0-11 arrive at the 4th CTA sync point
  sync_cta_unaligned();
}
/*===---------------------------------------------------------------------------------------------------------------===*/



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
static std::atomic<int> num_diffs(0);
constexpr static int MAX_NUM_DIFFS = 3;

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
      if (num_diffs < MAX_NUM_DIFFS) {
        if (num_diffs == 0) {
          LOG_INFO("Top %d diffs:\n", MAX_NUM_DIFFS);
        }
        LOG_INFO("[idx = %10d] cpu != gpu: %.5f != %.5f (abs diff = %.5f)\n", idx, cpu_data[idx], gpu_data[idx], diff);
        num_diffs++;
      }
    }
  }

  g_ok.fetch_add(ok);
  if (ng > 0) {
    g_ng.fetch_add(ng);
  }
}

template <typename T>
void print_abtype() {
  const char* func_sig = __PRETTY_FUNCTION__;
  char* type = const_cast<char*>(func_sig);
  char* ptr = type;
  int n = 0;
  while (*ptr != '=') {
    ptr++;
    n++;
  }

  type = type + (n + 2);
  ptr = type;

  n = 0;
  while (*ptr != ']') {
    ptr++;
    n++;
  }

  printf("ABType: %.*s\n", n, type);
}


#if defined(ABTYPE_F16)
using ABType = half_t;

extern "C" {
__global__ void blackwell_mmav5_f16_fp32(
  const __grid_constant__ CUtensorMap a_tensor_map,
  const __grid_constant__ CUtensorMap b_tensor_map,
  const __grid_constant__ CUtensorMap d_tensor_map,
  int M, int N, int K
) {
  tma_mmav5_fn<ABType, float, 64, 64, 64, 5>(a_tensor_map, b_tensor_map, d_tensor_map, M, N, K);
}
} // End extern "C"

static const char* kernel_name = "blackwell_mmav5_f16_fp32";
static auto kernel_ptr = blackwell_mmav5_f16_fp32;

#else
using ABType = bf16_t;

extern "C" {
__global__ void blackwell_mmav5_bf16_fp32(
  const __grid_constant__ CUtensorMap a_tensor_map,
  const __grid_constant__ CUtensorMap b_tensor_map,
  const __grid_constant__ CUtensorMap d_tensor_map,
  int M, int N, int K
) {
  tma_mmav5_fn<ABType, float, 64, 64, 64, 5>(a_tensor_map, b_tensor_map, d_tensor_map, M, N, K);
}
} // End extern "C"

static const char* kernel_name = "blackwell_mmav5_bf16_fp32";
static auto kernel_ptr = blackwell_mmav5_bf16_fp32;
#endif

using DType = float;


/*===---------------------------------------------------------------------------------------------------------------===*/
// main
/*===---------------------------------------------------------------------------------------------------------------===*/
int main(int argc, char** argv) {
  static_assert(std::is_same<ABType, half_t>() || std::is_same<ABType, bf16_t>(), "ABType is not supported!");
  print_abtype<ABType>();

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
  constexpr int K_TILE = 63;
  constexpr int NUM_STAGES = 5;

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
  CUtensorMap a_tensor_map{};
  bool status_A = create_tensor_map<ABType, 3>(&a_tensor_map, d_A_ptr, M, K, M_TILE, K_TILE); // BOX_COLS = 64
  if (!status_A) {
    return EXIT_FAILURE;
  }

  CUtensorMap b_tensor_map{};
  bool status_B = create_tensor_map<ABType, 3>(&b_tensor_map, d_B_ptr, N, K, N_TILE, K_TILE); // BOX_COLS = 64
  if (!status_B) {
    return EXIT_FAILURE;
  }

  CUtensorMap d_tensor_map{};
  bool status_D = create_tensor_map<DType, 3>(&d_tensor_map, d_D_ptr, M, N, M_TILE, N_TILE / 2);
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

  // Threadblock: 384 threads (12 warps)
  constexpr int THREADS_PER_BLOCK = 384;
  config.blockDim = dim3(THREADS_PER_BLOCK);
  config.stream = stream;

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  
  // For debugging:
  // std::uniform_int_distribution<int> dist(0, 3);

  std::uniform_real_distribution<float> dist(0.0, 0.1);

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
    std::size_t dyn_smem_bytes = NUM_STAGES * M_TILE * K_TILE * sizeof(ABType) + // a_stage_tiles
                                 NUM_STAGES * N_TILE * K_TILE * sizeof(ABType) + // b_stage_tiles
                                 M_TILE * N_TILE * sizeof(DType) +               // d_tile
                                 NUM_STAGES * sizeof(uint64_t) +                 // consumer_mbars
                                 NUM_STAGES * sizeof(uint64_t) +                 // producer_mbars
                                 sizeof(uint64_t) +                              // final_mbar
                                 sizeof(uint32_t) +                              // tmem_handle
                                 256 * sizeof(bool);

    config.dynamicSmemBytes = dyn_smem_bytes;

    CUDA_CHECK_ERROR(
      cudaFuncSetAttribute(
        kernel_ptr,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        dyn_smem_bytes
      )
    );

    // Launch GEMM kernel
    CUDA_CHECK_ERROR(
      cudaLaunchKernelEx(
        &config,
        kernel_ptr,
        a_tensor_map, b_tensor_map, d_tensor_map,
        M, N, K
      )
    );
    printf("Launched %s\n", kernel_name);

    // Copy output matrix
    cudaMemcpyAsync(h_D_gpu.data(), d_D_ptr, M * N * sizeof(DType), cudaMemcpyDeviceToHost, stream);

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());
    printf("Synchronize device done\n");

    // CPU
    printf("\nRun matmul_cpu_parallel\n");
    matmul_cpu_parallel<ABType, DType, M_TILE, N_TILE>(h_A.data(), h_B.data(), h_D_cpu.data(), stride_A, stride_B, stride_D, M, N, K);

    if (dump_data) {
      printf("Dumping cpu and gpu D matrices\n");
      const char indices_D[] = {'m', 'n'};
      fprint_mat(file_ptr, "h_D_cpu", h_D_cpu, indices_D, shape_D, stride_D);
      fprintf(file_ptr, "\n\n");
      fprint_mat(file_ptr, "h_D_gpu", h_D_gpu, indices_D, shape_D, stride_D);
    }

    // Verify
    printf("\nVerifying matrix D\n");
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
