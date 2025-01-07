#include <cstdio>
#include <bitset>
#include <random>

#include <cuda.h>
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
__device__ int canonical_warp_idx_sync() {
  return __shfl_sync(0xffffffff, threadIdx.x / kThreadsPerWarp, 0);
}

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

template <typename DType, typename ABType, int M, int N, int K>
__global__ void wgmma_kernel(
  DType* gD,
  ABType* gA,
  ABType* gB,
  const __grid_constant__ CUtensorMap tensor_map_A,
  const __grid_constant__ CUtensorMap tensor_map_B)
{
  __shared__ alignas(128) ABType sA[M * K];
  __shared__ alignas(128) ABType sB[N * K];
  __shared__ alignas(8) uint64_t mbar;

  constexpr int kTmaTransactionBytes = M * K * sizeof(ABType) + N * K * sizeof(ABType);
  int warp_idx = canonical_warp_idx_sync();
  int lane_predicate = elect_one_sync();

  // Initialize mbarrier
  if ((warp_idx == 0) && lane_predicate) {
    mbarrier_init(&mbar, 1);
  }

  // This is to ensure that `mbar` is really initialized across threads in the same cluster
  cluster_sync();

  // Initiate bulk tensor copies.
  if ((warp_idx == 0) && lane_predicate) {
    mbarrier_arrive_and_expect_tx(&mbar, kTmaTransactionBytes);

    cp_async_bulk_tensor_2d_global_to_shared(
      &sA[0],
      &tensor_map_A,
      0,
      0,
      &mbar
    );

    cp_async_bulk_tensor_2d_global_to_shared(
      &sB[0],
      &tensor_map_B,
      0,
      0,
      &mbar
    );
  }

  mbarrier_wait(&mbar, 0);

  // Create matrix descriptors
  uint64_t desc_a = create_desc<ABType>(&sA[0]);
  uint64_t desc_b = create_desc<ABType>(&sB[0]);

  int shape_d[] = {M, N};
  int stride_d[] = {N, 1};
  int* D_indexes = gen_indexes(threadIdx.x, stride_d);

  int idx00 = D_indexes[0];
  int idx01 = D_indexes[1];
  int idx02 = D_indexes[2];
  int idx03 = D_indexes[3];
  int idx04 = D_indexes[4];
  int idx05 = D_indexes[5];
  int idx06 = D_indexes[6];
  int idx07 = D_indexes[7];
  int idx08 = D_indexes[8];
  int idx09 = D_indexes[9];
  int idx10 = D_indexes[10];
  int idx11 = D_indexes[11];
  int idx12 = D_indexes[12];
  int idx13 = D_indexes[13];
  int idx14 = D_indexes[14];
  int idx15 = D_indexes[15];
  int idx16 = D_indexes[16];
  int idx17 = D_indexes[17];
  int idx18 = D_indexes[18];
  int idx19 = D_indexes[19];
  int idx20 = D_indexes[20];
  int idx21 = D_indexes[21];
  int idx22 = D_indexes[22];
  int idx23 = D_indexes[23];
  int idx24 = D_indexes[24];
  int idx25 = D_indexes[25];
  int idx26 = D_indexes[26];
  int idx27 = D_indexes[27];
  int idx28 = D_indexes[28];
  int idx29 = D_indexes[29];
  int idx30 = D_indexes[30];
  int idx31 = D_indexes[31];

  // Clear gD
  for (int i = 0; i < N/2; i++) {
    int idx = D_indexes[i];
    gD[idx] = DType(0);
  }

  // Make the generic proxy operations visible to the async proxy
  fence_proxy_async();

  // Enforce an ordering of register accesses between wgmma.mma_async and other operations.
  // `wgmma.fence` instruction establishes an ordering between prior accesses to any warpgroup registers 
  // and subsequent accesses to the same registers by a `wgmma.mma_async` instruction. 
  // Only the accumulator register and the input registers containing the fragments of matrix A require this ordering.
  warpgroup_arrive();

  // Issue WGMMA operation
  MMA_64x64x16_F32F16F16_SS mma_atom;
  mma_atom.fma(
    gD[idx00], gD[idx01], gD[idx02], gD[idx03], gD[idx04], gD[idx05], gD[idx06], gD[idx07],
    gD[idx08], gD[idx09], gD[idx10], gD[idx11], gD[idx12], gD[idx13], gD[idx14], gD[idx15],
    gD[idx16], gD[idx17], gD[idx18], gD[idx19], gD[idx20], gD[idx21], gD[idx22], gD[idx23],
    gD[idx24], gD[idx25], gD[idx26], gD[idx27], gD[idx28], gD[idx29], gD[idx30], gD[idx31],
    desc_a,
    desc_b
  );

  warpgroup_commit_batch();

  // Wait for MMA op to complete
  warpgroup_wait<0>();
}

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
  using ABType = bf16_t;

  constexpr int M = 64;
  constexpr int N = 64;
  constexpr int K = 16;
  
  int shape_a[] = {M, K};
  int stride_a[] = {K, 1};
  ABType* h_A_org = new ABType[M*K];
  ABType* h_A = new ABType[M*K];

  int shape_b[] = {K, N};
  int stride_b[] = {1, K};
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

    // Create tensor map
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
        wgmma_kernel<DType, ABType, M, N, K>,
        d_D,
        d_A,
        d_B,
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
