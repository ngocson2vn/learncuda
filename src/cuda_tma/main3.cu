#include "device.h"

#include <cuda_bf16.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "fprint_mat.h"

using bf16_t = nv_bfloat16;

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


/*===---------------------------------------------------------------------------------------------------------------===*/
// tma_kernel
/*===---------------------------------------------------------------------------------------------------------------===*/
template <typename DataType, int SMEM_ROWS, int SMEM_COLS>
__global__ void tma_kernel(const __grid_constant__ CUtensorMap tensor_map) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
  if (threadIdx.x == 0) printf("__CUDA_ARCH__: %d\n", __CUDA_ARCH__);
#endif

  static_assert(SMEM_ROWS * SMEM_COLS * sizeof(DataType) < 49152, "SMEM size exceeds the maximum value of 49152 bytes");
  assert(SMEM_ROWS * SMEM_COLS % blockDim.x == 0 && "SMEM size is not divisible by blockDim.x");

  printf("[tma_kernel] threadIdx.x %d starts; tensor_map: %p\n", threadIdx.x, &tensor_map);

  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) DataType smem_buffer[SMEM_ROWS * SMEM_COLS];
  // printf("smem_buffer: %ld\n", __cvta_generic_to_shared(smem_buffer));

  __shared__ uint64_t mbar;

  int col = 0;
  int row = blockIdx.x * SMEM_ROWS;
  int tid = threadIdx.x;

  if (tid == 0) {
    // mbarrier.init.shared::cta [addr], expected_tx;
    mbarrier_init(&mbar, 1);

    int expected_bytes = SMEM_ROWS * SMEM_COLS * sizeof(DataType);
    mbarrier_arrive_and_expect_tx_bytes(&mbar, expected_bytes);

    // Initiate bulk tensor copy.
    printf("[tma_kernel] threadIdx.x %d initiates bulk tensor copy\n", threadIdx.x);
    auto tmp_addr = __cvta_generic_to_shared(smem_buffer);
    printf("\nsmem_buffer addr: %lu\n\n", tmp_addr);

    cp_async_bulk_tensor_2d_global_to_shared(
      smem_buffer,
      &tensor_map,
      col,
      row,
      &mbar
    );
  }

  // Syncthreads to ensure that initialized barrier is visible to all threads.
  __syncthreads();

  // Wait for the data to transfer from global -> shared
  mbarrier_wait(&mbar, 1);
  printf("[tma_kernel] threadIdx.x %d arrived\n", threadIdx.x);

  // Symbolically modify a value in shared memory.
  // For each row, each thread operates on 1 element at column threadIdx.x
  for (int i = 0; i < SMEM_ROWS; i++) {
    int idx = SMEM_COLS * i + threadIdx.x;
    if (i == 0) {
      smem_buffer[idx] = blockIdx.x;
    } else {
      smem_buffer[idx] += blockIdx.x;
    }
  }

  // After syncthreads, writes by all threads are visible to TMA engine.
  fence_proxy_async();
  __syncthreads();

  // Initiate TMA transfer to copy shared memory to global memory
  if (tid == 0) {
    cp_async_bulk_tensor_2d_shared_to_global(&tensor_map, col, row, &smem_buffer);

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


template<typename DataType, int ROWS, int COLS, int BOX_ROWS, int BOX_COLS, bool rowMajor = true>
bool create_tensor_map(CUtensorMap* tensorMap, void* globalAddress) {
  // tensorRank is the number of dimensions of the array.
  constexpr uint32_t tensorRank = 2;

  // In CUDA, 
  // dim0 = column
  // dim1 = row
  uint64_t globalDim[tensorRank] = {COLS, ROWS};

  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  uint64_t ld = globalDim[0];
  if constexpr(!rowMajor) {
    ld = globalDim[1];
  }
  uint64_t globalStrides[tensorRank - 1] = {ld * sizeof(DataType)};
  printf("globalStrides[0] = %ld\n", globalStrides[0]);

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

int main(int argc, char** argv) {
  using DataType = bf16_t;
  constexpr int GMEM_ROWS = 64;
  constexpr int GMEM_COLS = 32;
  constexpr int SMEM_ROWS = 16;
  constexpr int SMEM_COLS = 32;
  constexpr int kDataSize = GMEM_ROWS * GMEM_COLS;
  constexpr int shape[] = {GMEM_ROWS, GMEM_COLS};
  constexpr int stride[] = {GMEM_COLS, 1};

  // Create matrix A as a host vector
  thrust::host_vector<DataType> h_data(kDataSize);

  // Initialize matrix A with zeros
  int idx = 0;
  for (int i = 0; i < GMEM_ROWS; i++) {
    for (int j = 0; j < GMEM_COLS; j++) {
      idx = i * stride[0] + j * stride[1];
      h_data[idx] = DataType(j);
    }
  }

  // Dump matrix A to output.txt
  const char* output_file = "output.txt";
  FILE* file_ptr = get_file_ptr(output_file);
  fprint_mat(file_ptr, "h_data", h_data.data(), shape, stride);
  fprintf(file_ptr, "\n\n");

  // Initialize GPU 0
  device_init(0);

  // Transfer matrix A from host to global memory on device
  thrust::device_vector<DataType> d_data = h_data;

  // Get the global memory pointer to matrix A on device
  void* gmem_ptr = d_data.data().get();
  printf("gmem_ptr: %p\n", gmem_ptr);

  // Create tensor map
  CUtensorMap tensor_map{};
  bool ok = create_tensor_map<DataType, GMEM_ROWS, GMEM_COLS, SMEM_ROWS, SMEM_COLS, true>(&tensor_map, gmem_ptr);
  if (!ok) {
    return EXIT_FAILURE;
  }

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

    /*
    Launch 4 block of TMA kernel, each block operates on a quarter of matrix A along column dimension
    |----------------------------|
    |     1                      |
    |----------------------------|
    |     2                      |
    |----------------------------|
    |     3                      |
    |----------------------------|
    |     4                      |
    |----------------------------|
    */
    // The grid dimension is not affected by cluster launch, and is still enumerated using number of blocks.
    // The grid dimension should be a multiple of cluster size.
    config.gridDim = dim3(4);
    config.blockDim = dim3(32);

    // Launch kernel
    CUDA_CHECK_ERROR(cudaLaunchKernelEx(&config, tma_kernel<DataType, SMEM_ROWS, SMEM_COLS>, tensor_map));
  }

  // Transfer matrix A from global memory to host memory
  thrust::host_vector<DataType> h_result = d_data;
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  // Dump matrix A to output.txt again
  fprint_mat(file_ptr, "h_result", h_result.data(), shape, stride);
  fprintf(file_ptr, "\n\n");

  printf("DONE\n");
  printf("Please check %s\n", output_file);
  return 0;
}
