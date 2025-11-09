#include <cstdio>
#include <bitset>
#include <random>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include "fprint_mat.h"

using bf16_t = nv_bfloat16;

#define STRINGIFY_TYPE(type) #type
#define TYPE_TO_STR(type) STRINGIFY_TYPE(type)

#define CUDA_CHECK_ERROR(e)                                    \
do {                                                           \
  cudaError_t code = (e);                                      \
  if (code != cudaSuccess) {                                   \
    fprintf(stderr, "<%s:%d> %s:\n    %s: %s\n",               \
            __FILE__, __LINE__, #e,                            \
            cudaGetErrorName(code), cudaGetErrorString(code)); \
    fflush(stderr);                                            \
    exit(1);                                                   \
  }                                                            \
} while (0)

void device_init(int device_id, bool quiet = false) {
  cudaDeviceProp device_prop;
  std::size_t    device_free_physmem;
  std::size_t    device_total_physmem;

  CUDA_CHECK_ERROR(cudaSetDevice(device_id));
  CUDA_CHECK_ERROR(cudaMemGetInfo(&device_free_physmem, &device_total_physmem));
  CUDA_CHECK_ERROR(cudaGetDeviceProperties(&device_prop, device_id));

  if (device_prop.major < 1) {
    fprintf(stderr, "Device does not support CUDA.\n");
    exit(1);
  }

  if (!quiet) {
    printf("Using device %d: %s  (SM%d, %d SMs)\n",
           device_id, device_prop.name,
           device_prop.major * 10 + device_prop.minor,
           device_prop.multiProcessorCount);
    fflush(stdout);
  }
}


constexpr static int kThreadsPerWarp = 32;

/// Returns a warp-uniform value indicating the canonical warp index of the calling threads.
/// Threads within the warp must be converged.
// __device__ int canonical_warp_idx_sync() {
//   return __shfl_sync(0xffffffff, threadIdx.x / kThreadsPerWarp, 0);
// }

// Elect one thread in the warp. 
// The elected thread gets its predicate set to true, all others obtain false.
__device__ uint32_t elect_one_sync()
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

__device__ void cluster_arrive()
{
  asm volatile("barrier.cluster.arrive.aligned;\n" : : );
}

__device__ void cluster_wait()
{
  asm volatile("barrier.cluster.wait.aligned;\n" : : );
}

__device__ void cluster_sync()
{
  cluster_arrive();
  cluster_wait();
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

__device__ void mbarrier_init(void const* smem_ptr, uint32_t arrive_count)
{
  uint64_t smem_addr = reinterpret_cast<uint64_t>(smem_ptr);
  asm volatile(
    "{\n\t"
      "mbarrier.init.shared::cta.b64 [%0], %1; \n"
    "}"
    :
    : "l"(smem_addr), "r"(arrive_count)
  );
}

// Performs an arrive operation + expected transaction bytes increment
__device__ void mbarrier_arrive_and_expect_tx(void const* smem_ptr, uint32_t transaction_bytes) {
  uint64_t smem_addr = reinterpret_cast<uint64_t>(smem_ptr);
  asm volatile(
    "{\n\t"
      "mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1; \n\t"
    "}"
    :
    : "l"(smem_addr), "r"(transaction_bytes)
  );
}

__device__ void mbarrier_wait(void const* smem_ptr, uint32_t phase) {
  uint64_t smem_addr = reinterpret_cast<uint64_t>(smem_ptr);
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
    : "l"(smem_addr), "r"(phase), "r"(ticks)
  );
}

__device__ inline void fence_proxy_async() {
  asm volatile("fence.proxy.async.shared::cta;" : : : "memory");
}

__device__ inline void cp_async_bulk_tensor_2d_global_to_shared(void* dst, const void* tensor_map, int x, int y, void* smem_bar) {
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
         :
         : "l"(reinterpret_cast<uint64_t>(dst)),
           "l"(reinterpret_cast<uint64_t>(tensor_map)),
           "r"(x),
           "r"(y),
           "l"(reinterpret_cast<uint64_t>(smem_bar))
         : "memory"
    );
}

__device__ inline void cp_async_bulk_tensor_2d_shared_to_global(const void* tensor_map, int x, int y, const void* smem_ptr) {
    asm volatile(
      "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
         :
         : "l"(reinterpret_cast<uint64_t>(tensor_map)), "r"(x), "r"(y)
           "l"(reinterpret_cast<uint64_t>(smem_ptr))
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
struct MMA_64x8x16_F32F16F16_SS
{
  __device__ static void
  fma(
      float& d00, float& d01, float& d02, float& d03,
      uint64_t const& desc_a,
      uint64_t const& desc_b)
  {
    asm volatile(
      "{\n"
        "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 "
        "{ %0,  %1,  %2,  %3 },"
        " %4,"
        " %5,"
        " %6, %7, %8, %9, %10;\n"
      "}\n"
        : "+f"(d00), "+f"(d01), "+f"(d02), "+f"(d03)
        :  "l"(desc_a), "l"(desc_b),
           "n"(scaleD), "n"(scaleA), "n"(scaleB), "n"(tnspA), "n"(tnspB)
    );
  }
};


template <int32_t scaleD = 1, int32_t scaleA = 1, int32_t scaleB = 1, int32_t tnspA = 0, int32_t tnspB = 0>
struct MMA_64x64x16_F32F16F16_SS
{
  __device__ static void
  fma(
      float& d00, float& d01, float& d02, float& d03, float& d04, float& d05, float& d06, float& d07,
      float& d08, float& d09, float& d10, float& d11, float& d12, float& d13, float& d14, float& d15,
      float& d16, float& d17, float& d18, float& d19, float& d20, float& d21, float& d22, float& d23,
      float& d24, float& d25, float& d26, float& d27, float& d28, float& d29, float& d30, float& d31,
      uint64_t const& desc_a,
      uint64_t const& desc_b)
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
        :  "l"(desc_a), "l"(desc_b),
           "n"(scaleD), "n"(scaleA), "n"(scaleB), "n"(tnspA), "n"(tnspB)
    );
  }
};


template<typename DataType, int ROWS, int COLS, bool row_major = true>
void create_tensor_map(CUtensorMap* tensor_map, DataType* gmem_ptr) {
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;

  uint64_t size[rank] = {COLS, ROWS};
  if constexpr(!row_major) {
    size[0] = ROWS;
    size[1] = COLS;
  }

  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {size[0] * sizeof(DataType)};
  printf("stride[0] = %ld\n", stride[0]);

  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {COLS, ROWS};
  if constexpr(!row_major) {
    box_size[0] = ROWS;
    box_size[1] = COLS;
  }

  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  CUtensorMapDataType dtype;
  if constexpr(std::is_same<DataType, half>()) {
    dtype = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
  } else if constexpr(std::is_same<DataType, bf16_t>()) {
    dtype = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
  } else if constexpr(std::is_same<DataType, float>()) {
    dtype = CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  } else {
    fprintf(stderr, "%s:%d: create_tensor_map does not support the provided DataType.\n", __FILE__, __LINE__);
    std::terminate();
  }

  printf("CUtensorMapDataType: %d\n", dtype);

  // Create the tensor descriptor.
  CUresult ret = cuTensorMapEncodeTiled(
    tensor_map,                 // CUtensorMap *tensorMap,
    dtype,
    rank,                       // cuuint32_t tensorRank,
    gmem_ptr,                   // void *globalAddress,
    size,                       // const cuuint64_t *globalDim,
    stride,                     // const cuuint64_t *globalStrides,
    box_size,                   // const cuuint32_t *boxDim,
    elem_stride,                // const cuuint32_t *elementStrides,

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
    fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, "Failed to create CUtensorMap object\n");
    std::terminate();
  }
}


template<typename DataType>
__device__ uint64_t create_desc(DataType* smem_ptr) {
  uint64_t desc = 0;

  // matrix-descriptor-encode(x)
  auto mde = [](uint64_t x) -> uint64_t { return ((x & 0x3FFFF) >> 0x4); };

  uint64_t smem_addr = mde(reinterpret_cast<uint64_t>(smem_ptr));

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
__device__ int* gen_indexes(int thread_idx, int* stride_d) {
  int c = stride_d[0] / 2;
  int* D_indexes = new int[c];

  int base_row = (thread_idx / 4) + (thread_idx / 32) * 8;
  int base_col  = (thread_idx % 4) * 2;
  
  int k = 0;
  for (int i = 0; i < (stride_d[0] / 8); i++) {
    D_indexes[k]   = base_row * stride_d[0] + (base_col + i * 8) * stride_d[1];
    D_indexes[k+1] = D_indexes[k] + 1;

    D_indexes[k+2] = (base_row + 8) * stride_d[0] + (base_col + i * 8) * stride_d[1];
    D_indexes[k+3] = D_indexes[k+2] + 1;
    k += 4;
  }

  return D_indexes;
}

/*===---------------------------------------------------------------------------------------------------------------===*/
// gemm_tma_wgmma_fp16_fp32
/*===---------------------------------------------------------------------------------------------------------------===*/
// Threadblock: 128 threads (4 warps) for one warp-group
constexpr int THREADS_PER_BLOCK = 128;
// Shared memory double buffering for A and B tiles
// We'll store A_smem shape [M_TILE, K_TILE] as FP16 (row-major)
// and B_smem shape [K_TILE, N_TILE] as FP16 (col-major in smem to match WGMMA flavor)
struct TmaDesc {
  uint64_t desc[8]; // opaque TMA descriptor storage (size per PTX ISA); aligned
};

// Kernel uses TMA mbarriers (64B) and smem
// __launch_bounds__(THREADS_PER_BLOCK, 2)
template <uint M_TILE=128, uint N_TILE=128, uint K_TILE=64> 
__global__
void gemm_tma_wgmma_fp16_fp32(
  float* __restrict__ D,       // [M, N], row-major
  const half* __restrict__ A,  // [M, K], row-major
  const half* __restrict__ B,  // [K, N], row-major (we'll transpose via TMA mapping to col-major in smem)
  int M, int N, int K,
  const __grid_constant__ TmaDesc* __restrict__ tmaA, // TMA descriptors for A
  const __grid_constant__ TmaDesc* __restrict__ tmaB  // TMA descriptors for B
) {
#if __CUDA_ARCH__ < 900
  return; // requires SM90
#endif
  // Shared memory layout:
  // [ mbarA(64B) | mbarB(64B) | A[2][M_TILE*K_TILE] | B[2][K_TILE*N_TILE] ]
  extern __shared__ uint8_t smem[];
  uint8_t* ptr = smem;

  // two mbarriers: one for A, one for B
  uint64_t* mbarA = reinterpret_cast<uint64_t*>(ptr);
  ptr += 64;
  uint64_t* mbarB = reinterpret_cast<uint64_t*>(ptr);
  ptr += 64;
  
  // Double buffers
  half* Asmem = reinterpret_cast<half*>(ptr);
  size_t Asz = size_t(M_TILE) * K_TILE; // elements per buffer
  ptr += 2 * Asz * sizeof(half);
  half* Bsmem = reinterpret_cast<half*>(ptr);
  size_t Bsz = size_t(K_TILE) * N_TILE;
  // ptr += 2 * Bsz * sizeof(half); // end

  // Compute block tile indices
  int block_m = blockIdx.y;
  int block_n = blockIdx.x;
  int m0 = block_m * M_TILE;
  int n0 = block_n * N_TILE;

  // Bounds check for full tiles (simple version)
  if (m0 + M_TILE > M || n0 + N_TILE > N) {
    // For simplicity, skip partial tiles in this didactic example.
    return;
  }

  // Form warp and lane info
  int tid = threadIdx.x;
  int warp_id = tid >> 5;
  int lane_id = tid & 31;

  // Initialize mbarriers once per block
  if (tid == 0) {
    // mbarrier.init.shared::cta [addr], expected_tx;
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(mbarA), "r"(1));
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" :: "r"(mbarB), "r"(1));
  }
  __syncthreads();

  // Accumulator fragment in registers. Each warp contributes a slice.
  // We will use wgmma 128x128x64.f16.f16.f32 variant; each thread holds some registers.
  // Reserve enough registers to hold per-thread accumulators; we’ll zero them.
  float acc[8]; // placeholder: number depends on mapping; keep small to compile
  #pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;

  // Helper lambdas for TMA launch
  auto tma_load_A = [&](int k_tile, int stage) {
    // Set up coordinates: (m, k) tile origin for A in global memory
    // Using TMA 2D: dims = [M_TILE, K_TILE] at global offsets (m0, k_tile)
    uint64_t desc = reinterpret_cast<const uint64_t&>(tmaA->desc[0]);

    // smem address for this stage
    half* dst = Asmem + stage * Asz;
    // Issue TMA cp.async.bulk.tensor.2d with mbarrier completion
    // PTX form (simplified):
    // cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes
    //   [smem_ptr], [desc], [coord_m, coord_k], [mbar], bytes;
    uint64_t smem_addr = static_cast<uint64_t>(__cvta_generic_to_shared(dst));
    int coord_m = m0;
    int coord_k = k_tile;

    // Bytes to transfer:
    int bytes = int(Asz * sizeof(half)); // full A tile
    asm volatile(
      "{\n"
      ".reg .b64 r_s, r_d, r_mb;\n"
      "mov.b64 r_s, %0;\n"
      "mov.b64 r_d, %1;\n"
      "mov.b64 r_mb, %2;\n"
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[r_s], [r_d, %3, %4], [r_mb], %5;\n"
      "}\n"
      :
      : "l"(smem_addr),
        "l"(tmaA->desc),
        "l"(mbarA),
        "r"(coord_m),
        "r"(coord_k),
        "r"(bytes)
      : "memory");
  };

  auto tma_load_B = [&](int k_tile, int stage) {
    // We want B in shared as [K_TILE, N_TILE] col-major to match wgmma flavor.
    // TMA descriptor for B will encode strides such that advancing first coord is k, second is n.
    uint64_t smem_addr = static_cast<uint64_t>(__cvta_generic_to_shared(Bsmem + stage * Bsz));
    int coord_k = k_tile;
    int coord_n = n0;
    int bytes = int(Bsz * sizeof(half));
    asm volatile(
      "{\n"
      ".reg .b64 r_s, r_d, r_mb;\n"
      "mov.b64 r_s, %0;\n"
      "mov.b64 r_d, %1;\n"
      "mov.b64 r_mb, %2;\n"
      "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes "
      "[r_s], [r_d, %3, %4], [r_mb], %5;\n"
      "}\n"
      :
      : "l"(smem_addr),
        "l"(tmaB->desc),
        "l"(mbarB),
        "r"(coord_k),
        "r"(coord_n),
        "r"(bytes)
      : "memory");
  };

  // Pipeline: prefetch stage 0
  int stages = 2;
  int stage = 0;

  // Arrive to mbarriers for first loads
  if (tid == 0) {
    // producer sets expected transaction count = 1 already in init, but we "arrive.expect_tx" before launch
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 [%0], %1;\n" :: "r"(mbarA), "r"(1));
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 [%0], %1;\n" :: "r"(mbarB), "r"(1));
  }
  __syncthreads();
  tma_load_A(0, stage);
  tma_load_B(0, stage);

  // Wait for both tiles to be ready before first compute
  // consumer_wait
  asm volatile("mbarrier.try_wait.parity.shared::cta.b64 %0, [%1];\n" : "=r"(stage) : "r"(mbarA));
  asm volatile("mbarrier.try_wait.parity.shared::cta.b64 %0, [%1];\n" : "=r"(stage) : "r"(mbarB));
  __syncthreads();

  // Main K loop with double buffering
  for (int k0 = 0; k0 < K; k0 += K_TILE) {
    int next_k = k0 + K_TILE;
    int next_stage = stage ^ 1;

    // Launch next TMA loads if there is a next slice
    if (next_k < K) {
      if (tid == 0) {
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 [%0], %1;\n" :: "r"(mbarA), "r"(1));
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 [%0], %1;\n" :: "r"(mbarB), "r"(1));
      }
      __syncwarp(); // not strictly necessary for all
      tma_load_A(next_k, next_stage);
      tma_load_B(next_k, next_stage);
    }

    // Issue WGMMA for current stage’s tiles in shared memory
    // Map current shared pointers for A and B
    uint64_t smemA = static_cast<uint64_t>(__cvta_generic_to_shared(Asmem + stage * Asz));
    uint64_t smemB = static_cast<uint64_t>(__cvta_generic_to_shared(Bsmem + stage * Bsz));

    // Each warp-group executes wgmma on 128x128x64 tile:
    // wgmma.mma_async.sync.aligned.m128n128k64.f32.f16.f16
    //   {acc...}, [smemA], [smemB], {acc...};
    // The exact register mapping is complex; here we demonstrate a minimal inline PTX call
    // using "descriptor" operands that point into shared memory. In practice, you will use
    // a per-thread fragment mapping. This snippet compiles but is schematic.
    asm volatile(
      "{\n"
      "  .reg .b64 a_desc, b_desc;\n"
      "  mov.b64 a_desc, %0;\n"
      "  mov.b64 b_desc, %1;\n"
      // The following is a conceptual call; real code must bind many accumulator regs.
      // We emulate compute by a no-op wgmma fence sequence for illustration.
      "  wgmma.fence.sync.aligned;\n"
      "  // Example: wgmma.mma_async.sync.aligned.m128n128k64.f32.f16.f16 {accumulators}, [a_desc], [b_desc], {accumulators};\n"
      "  // For a functional version, integrate CUTLASS-like register mapping.\n"
      "  wgmma.commit_group.sync.aligned;\n"
      "  wgmma.wait_group.sync.aligned %2;\n"
      "}\n"
      :
      : "l"(smemA), "l"(smemB), "n"(0)
      : "memory");

    // If next loads exist, wait for them to complete before swapping stage
    if (next_k < K) {
      int tmp;
      asm volatile("mbarrier.try_wait.parity.shared::cta.b64 %0, [%1];\n" : "=r"(tmp) : "r"(mbarA));
      asm volatile("mbarrier.try_wait.parity.shared::cta.b64 %0, [%1];\n" : "=r"(tmp) : "r"(mbarB));
      __syncthreads();
      stage = next_stage;
    }
  }

  // Epilogue: write accumulators to D
  // For demonstration, write zeros; in a real version, map acc registers to rows/cols.
  // Make each thread write a small stripe so kernel still produces defined output.
  int rows_per_thread = M_TILE / 32;  // simplistic partition
  int cols_per_warp = N_TILE / 4;     // 4 warps
  int row_base = m0 + lane_id * rows_per_thread;
  int col_base = n0 + warp_id * cols_per_warp;
  for (int i = 0; i < rows_per_thread; ++i) {
    for (int j = 0; j < cols_per_warp; ++j) {
      int r = row_base + i;
      int c = col_base + j;
      D[r * N + c] = 0.0f; // placeholder: should store from acc mapping
    }
  }
}
/*===---------------------------------------------------------------------------------------------------------------===*/



template <typename DType, typename ABType, int M, int N, int K>
DType* matmul_cpu(ABType* MatA, ABType* MatB) {
  DType* MatC = new DType[M * N];
  for (int i = 0; i < M; i++) {
    for (int j = 0; j < N; j++) {
      MatC[i * N + j] = 0;
      for (int k = 0; k < K; k++) {
        MatC[i * N + j] += static_cast<float>(MatA[i * K + k]) * static_cast<float>(MatB[k + j * K]);
      }
    }
  }

  return MatC;
}

int main(int argc, char** argv) {
  using DType = float;
  using ABType = half;

  constexpr int M = 128 * 4;
  constexpr int N = 128 * 4;
  constexpr int K = 64 * 4;
  
  int shape_a[] = {M, K};
  int stride_a[] = {K, 1};
  ABType* h_A_org = new ABType[M*K];
  ABType* h_A = new ABType[M*K];

  int shape_b[] = {K, N};
  int stride_b[] = {N, 1};
  ABType* h_B_org = new ABType[N*K];
  ABType* h_B = new ABType[N*K];

  std::random_device rd;  // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_int_distribution<int> dist(1, 10);


  ABType init_val = 1.0;
  for (int i = 0; i < shape_a[0]; i++) {
    for (int j = 0; j < shape_a[1]; j++) {
      h_A_org[i * stride_a[0] + j * stride_a[1]] = ABType(dist(gen));
    }
  }

  init_val = 0;
  for (int i = 0; i < shape_b[0]; i++) {
    for (int j = 0; j < shape_b[1]; j++) {
      h_B_org[i * stride_b[0] + j * stride_b[1]] = ABType(dist(gen));
    }
  }

  FILE* file_ptr = get_file_ptr("output.txt");
  fprint_mat(file_ptr, "h_A_org", h_A_org, shape_a, stride_a);
  fprintf(file_ptr, "\n\n");

  fprint_mat(file_ptr, "h_B_org", h_B_org, shape_b, stride_b);
  fprintf(file_ptr, "\n\n");

  DType* h_D_cpu = matmul_cpu<DType, ABType, M, N, K>(h_A_org, h_B_org);
  int shape_d[] = {M, N};
  int stride_d[] = {N, 1};
  fprint_mat(file_ptr, "h_D_cpu", h_D_cpu, shape_d, stride_d);
  fprintf(file_ptr, "\n\n");


  printf("sA layout:\n");
  for (int i = 0; i < M; i++) {
    if (i > 0 && i % 8 == 0) {
      printf("\n");
    }
    for (int j = 0; j < K; j++) {
      int idx = i * 8 + (j % 8) + (i / 8) * 64 + (j / 8) * 64;
      if (j > 0 && j % 8 == 0) {
        printf("  ");
      }
      printf("%4d ", idx);
      h_A[idx] = h_A_org[i * stride_a[0] + j * stride_a[1]];
    }
    printf("\n");
  }
  printf("\n\n");

  printf("sB layout:\n");
  for (int i = 0; i < K; i++) {
    if (i > 0 && i % 8 == 0) {
      printf("\n");
    }
    for (int j = 0; j < N; j++) {
      int idx = (i % 8) + (i / 8) * 64 + (j / 8) * 64 + j * 8;
      if (j > 0 && j % 8 == 0) {
        printf("  ");
      }
      printf("%4d ", idx);
      h_B[idx] = h_B_org[i * stride_b[0] + j * stride_b[1]];
      // printf("%10.5f", static_cast<float>(h_B[idx]));
    }
    printf("\n");
  }
  printf("\n");

  // Init cuda
  device_init(0);
  printf("\n");

  // Create a stream
  cudaStream_t stream;
  CUDA_CHECK_ERROR(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  // Transfer data from host to device
  ABType* d_A;
  cudaMalloc((void**)&d_A, M * K * sizeof(ABType));
  cudaMemcpyAsync(d_A, h_A, M * K * sizeof(ABType), cudaMemcpyHostToDevice, stream);

  ABType* d_B;
  cudaMalloc((void**)&d_B, N * K * sizeof(ABType));
  cudaMemcpyAsync(d_B, h_B, N * K * sizeof(ABType), cudaMemcpyHostToDevice, stream);

  DType* d_D;
  cudaMalloc((void**)&d_D, M * N * sizeof(DType));

    // Create tensor maps
  CUtensorMap tensor_map_A{};
  create_tensor_map<ABType, M, K, true>(&tensor_map_A, d_A);

  CUtensorMap tensor_map_B{};
  create_tensor_map<ABType, K, N, false>(&tensor_map_B, d_B);

  // Kernel invocation with runtime cluster size
  {
    cudaLaunchConfig_t config = {0};
    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks. The grid dimension should be a multiple of cluster size.
    config.gridDim = dim3(1);
    config.blockDim = dim3(128);
    config.stream = stream;

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 1; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    // Launch wgmma_kernel
    CUDA_CHECK_ERROR(
      cudaLaunchKernelEx(
        &config,
        gemm_tma_wgmma_fp16_fp32<128, 128, 64>,
        d_D,
        d_A,
        d_B,
        M, N, K,
        tensor_map_A,
        tensor_map_B
      )
    );
  }

  // Copy output matrix
  DType* h_D_gpu = new DType[M * N];
  cudaMemcpyAsync(h_D_gpu, d_D, M * N * sizeof(DType), cudaMemcpyDeviceToHost, stream);
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  fprint_mat(file_ptr, "h_D_gpu", h_D_gpu, shape_d, stride_d);

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
