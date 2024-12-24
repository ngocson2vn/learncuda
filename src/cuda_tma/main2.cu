/*
Ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-tma-to-transfer-multi-dimensional-arrays
*/

#include <cuda.h>
#include <cuda/barrier>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "fprint_mat.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;

#define CUDA_CHECK_ERROR(e)                                    \
do {                                                           \
  cudaError_t code = cudaGetLastError();                       \
  if (code != cudaSuccess) {                                   \
    fprintf(stderr, "<%s:%d> last error:\n    %s: %s\n",       \
            __FILE__, __LINE__,                                \
            cudaGetErrorName(code), cudaGetErrorString(code)); \
    fflush(stderr);                                            \
    exit(1);                                                   \
  }                                                            \
  code = (e);                                                  \
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


template <int SMEM_HEIGHT = 32, int SMEM_WIDTH = 32, typename DataType>
__global__ void tma_kernel(const __grid_constant__ CUtensorMap tensor_map, int x, int y) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
  if (threadIdx.x == 0) printf("__CUDA_ARCH__: %d\n", __CUDA_ARCH__);
#endif

  static_assert(SMEM_HEIGHT * SMEM_WIDTH * sizeof(DataType) < 49152, "SMEM size exceeds the maximum value of 49152 bytes");
  assert(SMEM_HEIGHT * SMEM_WIDTH % blockDim.x == 0 && "SMEM size is not divisible by blockDim.x");

  printf("[tma_kernel] threadIdx.x %d starts; tensor_map: %p\n", threadIdx.x, &tensor_map);

  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) DataType smem_buffer[SMEM_HEIGHT * SMEM_WIDTH];
  // printf("smem_buffer: %ld\n", reinterpret_cast<uint64_t>(smem_buffer));

  // Initialize shared memory barrier with the number of threads participating in the barrier.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;

  if (threadIdx.x == 0) {
    // Initialize barrier. All `blockDim.x` threads in block participate.
    init(&bar, blockDim.x);

    // Make initialized barrier visible in async proxy.
    fence_proxy_async();
  }

  // Syncthreads so initialized barrier is visible to all threads.
  __syncthreads();

  barrier::arrival_token token;
  if (threadIdx.x == 0) {
    // Initiate bulk tensor copy.
    printf("[tma_kernel] threadIdx.x %d initiates bulk tensor copy\n", threadIdx.x);
    cp_async_bulk_tensor_2d_global_to_shared(
      smem_buffer,
      &tensor_map,
      x,
      y,
      cuda::device::barrier_native_handle(bar)
    );

    // Arrive on the barrier and tell how many bytes are expected to come in.
    token = cuda::device::barrier_arrive_tx(bar, 1, sizeof(smem_buffer));
  } else {
    // Other threads just arrive.
    token = bar.arrive();
  }

  // Wait for the data to have arrived.
  bar.wait(std::move(token));
  printf("[tma_kernel] threadIdx.x %d arrived\n", threadIdx.x);

  // Symbolically modify a value in shared memory.
  const int k = SMEM_HEIGHT * SMEM_WIDTH / blockDim.x;
  for (int i = 0; i < k; i++) {
    smem_buffer[threadIdx.x * k + i] = threadIdx.x + y;
  }

  // Wait for shared memory writes to be visible to TMA engine.
  fence_proxy_async();
  __syncthreads(); // After syncthreads, writes by all threads are visible to TMA engine.

  // Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, x, y, &smem_buffer);

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


int main(int argc, char** argv) {
  constexpr int GMEM_HEIGHT = 64;
  constexpr int GMEM_WIDTH = 64;
  constexpr int SMEM_HEIGHT = 32;
  constexpr int SMEM_WIDTH = 32;
  constexpr int kDataSize = GMEM_HEIGHT * GMEM_WIDTH;

  // Create matrix A as a host vector
  thrust::host_vector<int> h_data(kDataSize);

  // Initialize matrix A with zeros
  for (int i = 0; i < kDataSize; i++) h_data[i] = 0;

  // Dump matrix A to output.txt
  FILE* file_ptr = get_file_ptr("output.txt");
  fprint_mat(file_ptr, "h_data", h_data.data(), dim3(GMEM_HEIGHT, GMEM_WIDTH, 1));
  fprintf(file_ptr, "\n\n");

  // Initialize GPU 0
  device_init(0);

  // Transfer matrix A from host to global memory on device
  thrust::device_vector<int> d_data = h_data;

  // Get the global memory pointer to matrix A on device
  void* gmem_ptr = d_data.data().get();
  printf("gmem_ptr: %p\n", gmem_ptr);

  // Create tensor map
  CUtensorMap tensor_map{};

  // rank is the number of dimensions of the array.
  constexpr uint32_t rank = 2;

  uint64_t size[rank] = {GMEM_WIDTH, GMEM_HEIGHT};

  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};

  // The box_size is the size of the shared memory buffer that is used as the
  // destination of a TMA transfer.
  uint32_t box_size[rank] = {SMEM_HEIGHT, SMEM_WIDTH};

  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  uint32_t elem_stride[rank] = {1, 1};

  // Create the tensor descriptor.
  CUresult ret = cuTensorMapEncodeTiled(
    &tensor_map,                // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
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
    return 1;
  }

  // Kernel invocation with runtime cluster size
  {
    cudaLaunchConfig_t config = {0};
    // The grid dimension is not affected by cluster launch, and is still enumerated
    // using number of blocks.
    // The grid dimension should be a multiple of cluster size.
    config.gridDim = dim3(1);
    config.blockDim = dim3(32);

    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 1; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;
    config.attrs = attribute;
    config.numAttrs = 1;

    /*
    Launch 4 TMA kernels, each kernel operates on a quarter of matrix A
    | 1 | 3 |
    |---|---|
    | 2 | 4 |
    */

    // Launch kernel 1
    int x = 0;
    int y = 0;
    CUDA_CHECK_ERROR(cudaLaunchKernelEx(&config, tma_kernel<SMEM_HEIGHT, SMEM_WIDTH, int>, tensor_map, x, y));

    // Launch kernel 2
    x = SMEM_HEIGHT;
    CUDA_CHECK_ERROR(cudaLaunchKernelEx(&config, tma_kernel<SMEM_HEIGHT, SMEM_WIDTH, int>, tensor_map, x, y));

    // Launch kernel 3
    x = 0;
    y = SMEM_WIDTH;
    CUDA_CHECK_ERROR(cudaLaunchKernelEx(&config, tma_kernel<SMEM_HEIGHT, SMEM_WIDTH, int>, tensor_map, x, y));

    // Launch kernel 4
    x = SMEM_HEIGHT;
    y = SMEM_WIDTH;
    CUDA_CHECK_ERROR(cudaLaunchKernelEx(&config, tma_kernel<SMEM_HEIGHT, SMEM_WIDTH, int>, tensor_map, x, y));
  }

  // Transfer matrix A from global memory to host memory
  thrust::host_vector<int> h_result = d_data;
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  // Dump matrix A to output.txt again
  fprint_mat(file_ptr, "h_result", h_result.data(), dim3(GMEM_HEIGHT, GMEM_WIDTH, 1));
  fprintf(file_ptr, "\n\n");

  printf("DONE\n");
  return 0;
}
