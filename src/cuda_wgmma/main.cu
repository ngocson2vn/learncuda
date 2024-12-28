#include <cstdio>
#include <bitset>

#include <cuda.h>
#include <cuda_bf16.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "fprint_mat.h"

using bf16_t = __nv_bfloat16;

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
      ".reg .pred %%px;\n"
      "elect.sync _|%%px, %1;\n"
      "@%%px mov.s32 %0, 1;\n"
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

template <int32_t scaleA = 1, int32_t scaleB = 1, int32_t tnspA = 1, int32_t tnspB = 1>
struct MMA_64x64x16_F16F16F16_SS
{
  __device__ static void
  fma(uint64_t const& desc_a,
      uint64_t const& desc_b,
      uint32_t      & d00, uint32_t      & d01, uint32_t      & d02, uint32_t      & d03,
      uint32_t      & d04, uint32_t      & d05, uint32_t      & d06, uint32_t      & d07,
      uint32_t      & d08, uint32_t      & d09, uint32_t      & d10, uint32_t      & d11,
      uint32_t      & d12, uint32_t      & d13, uint32_t      & d14, uint32_t      & d15,
      uint32_t const scale_D = 1)
  {
    asm volatile(
    "{\n"
      ".reg .pred p;\n"
      "setp.ne.b32 p, %18, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16 "
      "{%0,  %1,  %2,  %3,  %4,  %5,  %6,  %7,  "
      " %8,  %9,  %10, %11, %12, %13, %14, %15},"
      " %16,"
      " %17,"
      " p,   %19, %20, %21, %22;\n"
    "}\n"
      : "+r"(d00), "+r"(d01), "+r"(d02), "+r"(d03),
        "+r"(d04), "+r"(d05), "+r"(d06), "+r"(d07),
        "+r"(d08), "+r"(d09), "+r"(d10), "+r"(d11),
        "+r"(d12), "+r"(d13), "+r"(d14), "+r"(d15)
      :  "l"(desc_a),
         "l"(desc_b),
         "r"(int32_t(scale_D)), "n"(int32_t(scaleA)), "n"(int32_t(scaleB)), "n"(int32_t(tnspA)), "n"(int32_t(tnspB)));
  }
};


template <typename DataType, int M = 64, int N = 64, int K = 16>
__global__ void wgmma_kernel(
  DataType       *D,
  const __grid_constant__ CUtensorMap tensor_map_A,
  const __grid_constant__ CUtensorMap tensor_map_B)
{
  __shared__ alignas(128) DataType sA[M * K];
  __shared__ alignas(128) DataType sB[N * K];
  __shared__ alignas(8) uint64_t mbar;

  constexpr int kTmaTransactionBytes = sizeof(DataType) * (M * K + N * K);
  int warp_idx = canonical_warp_idx_sync();
  int lane_predicate = elect_one_sync();

  // Initialize mbarrier
  if ((warp_idx == 0) && lane_predicate) {
    mbarrier_init(&mbar, 1);
  }

  // This is to ensure that `mbar` is really initialized across threads in the same cluster
  cluster_sync();

  // Initiate bulk tensor copy.
  if ((warp_idx == 0) && lane_predicate) {
    printf("[wgmma_kernel] threadIdx.x %d initiates bulk tensor copies\n", threadIdx.x);

    mbarrier_arrive_and_expect_tx(&mbar, kTmaTransactionBytes);

    cp_async_bulk_tensor_2d_global_to_shared(
      sA,
      &tensor_map_A,
      0,
      0,
      &mbar
    );

    cp_async_bulk_tensor_2d_global_to_shared(
      sB,
      &tensor_map_B,
      0,
      0,
      &mbar
    );
  }

  mbarrier_wait(&mbar, 0);
  printf("threadIdx.x %d done\n", threadIdx.x);

  // Issue Tensor Core operation
  // MMA_64x64x16_F16F16F16_SS::kernel_wgmma(

  // );
}

template<
  int GMEM_HEIGHT = 64,
  int GMEM_WIDTH = 16,
  bool is_row_major = true
>
void create_tensor_map(CUtensorMap* tensor_map, void* gmem_ptr) {
  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;

  uint64_t size[rank] = {0, 0};
  if constexpr(is_row_major) {
    size[0] = GMEM_WIDTH;
    size[1] = GMEM_HEIGHT;
  } else {
    size[0] = GMEM_HEIGHT;
    size[1] = GMEM_WIDTH;
  }

  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {0};
  stride[0] = {size[0] * sizeof(bf16_t)};

  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {GMEM_HEIGHT, GMEM_WIDTH};

  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Create the tensor descriptor.
  CUresult ret = cuTensorMapEncodeTiled(
    tensor_map,                 // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,
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
    printf("Failed to create CUtensorMap object\n");
    std::terminate();
  }
}

int main(int argc, char** argv) {
  int m = 64;
  int n = 64;
  int k = 16;
  thrust::host_vector<bf16_t> h_A(m*k);
  thrust::host_vector<bf16_t> h_B(n*k);
  thrust::host_vector<bf16_t> h_D(m*n);

  // Initialize the tensors
  bf16_t val{1.0};
  // int32_t rep = val;
  // for (int i = 0; i < 3; i++) {
  //   rep = rep << 8;
  //   val = val | rep;
  // }
  // fprintf(file_ptr, "val: %s\n\n", std::bitset<32>(val).to_string().c_str());

  for (int j = 0; j < m*k; ++j) h_A[j] = bf16_t(val);
  for (int j = 0; j < n*k; ++j) h_B[j] = bf16_t(val);
  for (int j = 0; j < m*n; ++j) h_D[j] = bf16_t(0);

  FILE* file_ptr = get_file_ptr("output.txt");
  fprint_mat(file_ptr, "h_A", h_A.data(), dim3(m, k, 1));
  fprintf(file_ptr, "\n\n");

  fprint_mat(file_ptr, "h_B", h_B.data(), dim3(k, n, 1));
  fprintf(file_ptr, "\n\n");

  // Init cuda
  device_init(0);

  thrust::device_vector<bf16_t> d_A = h_A;
  thrust::device_vector<bf16_t> d_B = h_B;
  thrust::device_vector<bf16_t> d_D = h_D;

    // Create tensor map
  CUtensorMap tensor_map_A{};
  create_tensor_map(&tensor_map_A, d_A.data().get());
  CUtensorMap tensor_map_B{};
  create_tensor_map(&tensor_map_B, d_B.data().get());

  // Kernel invocation with runtime cluster size
  {
    cudaLaunchConfig_t config = {0};
    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension should be a multiple of cluster size.
    config.gridDim = dim3(1);
    config.blockDim = dim3(128);

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
        wgmma_kernel<bf16_t>,
        d_D.data().get(),
        tensor_map_A,
        tensor_map_B
      )
    );
  }

  // Copy output matrix
  // thrust::host_vector<bf16_t> h_result = d_D;
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  // fprintf(file_ptr, "\n\n");
  // fprint_mat(file_ptr, "h_D", h_D.data(), dim3(m, n, 1));
}
