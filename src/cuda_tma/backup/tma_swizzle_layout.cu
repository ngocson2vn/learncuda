#include <mutex>
#include "device.h"

#include <cuda_bf16.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "fprint_mat.h"

using bf16_t = nv_bfloat16;

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
  fprintf(stderr, "ERROR ");                  \
  fprintf(stderr, format_str, ##__VA_ARGS__); \
} while(0)


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

__device__ inline void cp_async_bulk_tensor_2d_global_to_shared(void* smem_ptr, const void* tensorMap, int col, int row, void* smem_bar) {
    asm volatile(
      "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
         :
         : "l"(__cvta_generic_to_shared(smem_ptr)),
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
__global__ void tma_kernel(
  const __grid_constant__ CUtensorMap tensor_map_input_man,
  const __grid_constant__ CUtensorMap tensor_map_input_tma,
  const __grid_constant__ CUtensorMap tensor_map_output_man,
  const __grid_constant__ CUtensorMap tensor_map_output_tma) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 200)
  if (threadIdx.x == 0) printf("__CUDA_ARCH__: %d\n", __CUDA_ARCH__);
#endif

  static_assert(SMEM_ROWS * SMEM_COLS * sizeof(DataType) < 49152, "SMEM size exceeds the maximum value of 49152 bytes");
  assert(SMEM_ROWS * SMEM_COLS % blockDim.x == 0 && "SMEM size is not divisible by blockDim.x");

  // printf("[tma_kernel] threadIdx.x %d starts; tensor_map_input_man: %p\n", threadIdx.x, &tensor_map_input_man);

  // The destination shared memory buffer of a bulk tensor operation should be
  // 128 byte aligned.
  __shared__ alignas(128) DataType smem_buffer_tmp[SMEM_ROWS * SMEM_COLS];
  __shared__ alignas(128) DataType smem_buffer_man[SMEM_ROWS * SMEM_COLS];
  __shared__ alignas(128) DataType smem_buffer_tma[SMEM_ROWS * SMEM_COLS];

  __shared__ uint64_t mbar_man;
  __shared__ uint64_t mbar_tma;

  int col = 0;
  int row = 0;
  int tid = threadIdx.x;

  if (tid == 0) {
    // mbarrier.init.shared::cta [addr], expected_tx;
    mbarrier_init(&mbar_man, 1);
    mbarrier_init(&mbar_tma, 1);

    int expected_bytes = SMEM_ROWS * SMEM_COLS * sizeof(DataType);
    mbarrier_arrive_and_expect_tx_bytes(&mbar_man, expected_bytes);
    mbarrier_arrive_and_expect_tx_bytes(&mbar_tma, expected_bytes);

    // Initiate bulk tensor copy.
    cp_async_bulk_tensor_2d_global_to_shared(
      smem_buffer_tmp,
      &tensor_map_input_man,
      col,
      row,
      &mbar_man
    );

    cp_async_bulk_tensor_2d_global_to_shared(
      smem_buffer_tma,
      &tensor_map_input_tma,
      col,
      row,
      &mbar_tma
    );

    printf("[tma_kernel] threadIdx.x %d initiated bulk tensor copy\n", threadIdx.x);
  }

  // Syncthreads to ensure that initialized barrier is visible to all threads.
  __syncthreads();

  // Wait for the data to transfer from global -> shared
  mbarrier_wait(&mbar_man, 0);
  mbarrier_wait(&mbar_tma, 0);
  printf("[tma_kernel] threadIdx.x %d arrived\n", threadIdx.x);

  // This fence makes the loaded data globally visible to the CTA.
  // Only after this fence are the normal stores below guaranteed 
  // to read the up-to-date data from the SMEM buffer.
  fence_proxy_async();

  if (tid == 0) {
    for (int m = 0; m < SMEM_ROWS; m++) {
      printf("m = %d: ", m);
      for (int k = 0; k < SMEM_COLS; k++) {
        int idx = m * SMEM_COLS + k;
        printf("%7.2f", (float)smem_buffer_tmp[idx]);

        // Manually swizzle smem_buffer_man
        int linear_idx = m * 8 + (k % 8) + (m / 8) * 64 + (k / 8) * 64;
        smem_buffer_man[linear_idx] = smem_buffer_tmp[idx];
      }
      printf("\n");
    }
  }

  // Make smem_buffer_man and smem_buffer_tma visible to TMA engine
  fence_proxy_async();

  // Initiate TMA transfer to copy shared memory to global memory
  if (tid == 0) {
    // Manual
    cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_output_man, col, row, &smem_buffer_man);

    // TMA
    cp_async_bulk_tensor_2d_shared_to_global(&tensor_map_output_tma, col, row, &smem_buffer_tma);

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

template<typename DataType, CUtensorMapSwizzle swizzleMode>
bool create_tensor_map(CUtensorMap* tensorMap, void* globalAddress, uint ROWS, uint COLS, uint BOX_ROWS, uint BOX_COLS) {
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
    LOG_ERROR("%s:%d: create_tensor_map does not support the provided DataType.\n", __FILE__, __LINE__);
    return false;
  }

  LOG_DEBUG("CUtensorMapDataType: %d\n", tensorDataType);

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
    swizzleMode,

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


int main(int argc, char** argv) {
  using DataType = bf16_t;
  constexpr uint SMEM_ROWS = 64;
  constexpr uint SMEM_COLS = 16;
  
  int tile_count = 1;
  const uint GMEM_ROWS = SMEM_ROWS * tile_count;
  const uint GMEM_COLS = SMEM_COLS * tile_count;
  const uint kDataSize = GMEM_ROWS * GMEM_COLS;
  const uint shape[] = {GMEM_ROWS, GMEM_COLS};
  const uint stride[] = {GMEM_COLS, 1};

  // Create matrix A as a host vector
  thrust::host_vector<DataType> h_input(kDataSize, DataType(0));

  // Initialize matrix A with zeros
  int idx = 0;
  for (int i = 0; i < GMEM_ROWS; i++) {
    for (int j = 0; j < GMEM_COLS; j++) {
      idx = i * stride[0] + j * stride[1];
      h_input[idx] = DataType(j);
    }
  }

  // Dump matrix A to output.txt
  const char* output_file = "output.txt";
  FILE* file_ptr = get_file_ptr(output_file);
  fprint_mat(file_ptr, "h_input", h_input.data(), shape, stride);
  fprintf(file_ptr, "\n\n");

  // Initialize GPU 0
  device_init(0);

  // Transfer matrix A from host to global memory on device
  thrust::device_vector<DataType> d_input = h_input;
  thrust::device_vector<DataType> d_output_man(kDataSize, DataType(0));
  thrust::device_vector<DataType> d_output_tma(kDataSize, DataType(0));

  // Get the global memory pointer to matrix A on device
  void* gmem_input_ptr = d_input.data().get();
  printf("gmem_input_ptr: %p\n", gmem_input_ptr);

  // Create tensor maps
  CUtensorMap tensor_map_input_man{};
  bool status_input_man = create_tensor_map<DataType, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE>(
    &tensor_map_input_man, gmem_input_ptr, GMEM_ROWS, GMEM_COLS, SMEM_ROWS, SMEM_COLS);
  if (!status_input_man) {
    return EXIT_FAILURE;
  }

  CUtensorMap tensor_map_input_tma{};
  bool status_input_tma = create_tensor_map<DataType, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B>(
    &tensor_map_input_tma, gmem_input_ptr, GMEM_ROWS, GMEM_COLS, SMEM_ROWS, SMEM_COLS);
  if (!status_input_tma) {
    return EXIT_FAILURE;
  }

  // Manual swizzle
  void* gmem_output_ptr_man = d_output_man.data().get();
  printf("gmem_output_ptr_man: %p\n", gmem_output_ptr_man);

  CUtensorMap tensor_map_output_man{};
  bool status_output_man = create_tensor_map<DataType, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE>(
    &tensor_map_output_man, gmem_output_ptr_man, GMEM_ROWS, GMEM_COLS, SMEM_ROWS, SMEM_COLS);
  if (!status_output_man) {
    return EXIT_FAILURE;
  }

  // TMA swizzle
  void* gmem_ptr_tma = d_output_tma.data().get();
  printf("gmem_ptr_tma: %p\n", gmem_ptr_tma);

  CUtensorMap tensor_map_output_tma{};
  // CU_TENSOR_MAP_SWIZZLE_32B works but the layout is incorrect
  // bool tma_status = create_tensor_map<DataType, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_32B>(
  //   &tensor_map_output_tma, gmem_ptr_tma, GMEM_ROWS, GMEM_COLS, SMEM_ROWS, SMEM_COLS);

  // cudaErrorIllegalAddress: an illegal memory access was encountered
  bool status_output_tma = create_tensor_map<DataType, CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE>(
    &tensor_map_output_tma, gmem_ptr_tma, GMEM_ROWS, GMEM_COLS, SMEM_ROWS, SMEM_COLS);
  if (!status_output_tma) {
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

    // The grid dimension is not affected by cluster launch, and is still enumerated using number of blocks.
    // The grid dimension should be a multiple of cluster size.
    config.gridDim = dim3(1);
    config.blockDim = dim3(32);

    // Launch kernel
    CUDA_CHECK_ERROR(
      cudaLaunchKernelEx(
        &config, 
        tma_kernel<DataType, SMEM_ROWS, SMEM_COLS>, 
        tensor_map_input_man, 
        tensor_map_input_tma,
        tensor_map_output_man,
        tensor_map_output_tma
      )
    );
  }

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  // Transfer matrix A from global memory to host memory
  thrust::host_vector<DataType> h_result_man = d_output_man;
  thrust::host_vector<DataType> h_result_tma = d_output_tma;

  // Dump matrix A to output.txt again
  fprint_mat(file_ptr, "h_result_man", h_result_man.data(), shape, stride);
  fprintf(file_ptr, "\n\n");

  fprint_mat(file_ptr, "h_result_tma", h_result_tma.data(), shape, stride);
  fprintf(file_ptr, "\n\n");

  printf("DONE\n");
  printf("Please check %s\n", output_file);
  return 0;
}
