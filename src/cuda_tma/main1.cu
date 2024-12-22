/*
Ref: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#using-tma-to-transfer-one-dimensional-arrays
*/

#include <cuda/barrier>
#include <cuda/ptx>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace ptx = cuda::ptx;

static constexpr size_t buf_len = 1024;
__global__ void add_one_kernel(int* data)
{
  // Shared memory buffer. The destination shared memory buffer of
  // a bulk operations should be 16 byte aligned.
  __shared__ alignas(16) int smem_data[buf_len];

  // 1. a) Initialize shared memory barrier with the number of threads participating in the barrier.
  //    b) Make initialized barrier visible in async proxy.
  #pragma nv_diag_suppress static_var_with_dynamic_init
  __shared__ barrier bar;
  if (threadIdx.x == 0) { 
    init(&bar, blockDim.x);                      // a)
    ptx::fence_proxy_async(ptx::space_shared);   // b)
  }
  __syncthreads(); // Ensure that the barrier object is really initialized.

  // 2. Initiate TMA transfer to copy global to shared memory.
  if (threadIdx.x == 0) {
    // 3a. cuda::memcpy_async arrives on the barrier and communicates
    //     how many bytes are expected to come in (the transaction count)
    cuda::memcpy_async(
        smem_data, 
        data, 
        cuda::aligned_size_t<16>(sizeof(smem_data)),
        bar
    );
  }

  // 3b. All threads arrive on the barrier
  barrier::arrival_token token = bar.arrive();
  
  // 3c. Wait for the data to have arrived.
  bar.wait(std::move(token));

  // 4. Compute saxpy and write back to shared memory
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x) {
    smem_data[i] *= 2;
  }

  // 5. Wait for shared memory writes to be visible to TMA engine.
  ptx::fence_proxy_async(ptx::space_shared);   // b)
  __syncthreads();

  // After syncthreads, writes by all threads are visible to TMA engine.

  // 6. Initiate TMA transfer to copy shared memory to global memory
  if (threadIdx.x == 0) {
    ptx::cp_async_bulk(
        ptx::space_global,
        ptx::space_shared,
        data, smem_data, sizeof(smem_data));

    // 7. Wait for TMA transfer to have finished reading shared memory.
    // Create a "bulk async-group" out of the previous bulk copy operation.
    ptx::cp_async_bulk_commit_group();

    // Wait for the group to have completed reading from shared memory.
    ptx::cp_async_bulk_wait_group_read(ptx::n32_t<0>());
  }

  printf("threadIdx.x %d done\n", threadIdx.x);
}

int main(int argc, char** argv) {
  constexpr int kThreads = 32;
  constexpr int kDataSize = 1024;
  thrust::host_vector<int> h_data(kDataSize);
  for (int i = 0; i < kDataSize; i++) h_data[i] = i;

  printf("\n\n");
  printf("h_data: ");
  for (int i = 0; i < h_data.size() - 1; i++) {
    printf("%d ", h_data[i]);
  }
  printf("%d\n\n", h_data[h_data.size() - 1]);

  device_init(0);

  thrust::device_vector<int> d_data = h_data;
  dim3 grid_dim(1);
  dim3 block_dim(kThreads);
  add_one_kernel<<<grid_dim, block_dim>>>(d_data.data().get());
  
  thrust::host_vector<int> h_result = d_data;
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
  
  printf("\n\n");
  printf("h_result: ");
  for (int i = 0; i < h_result.size() - 1; i++) {
    printf("%d ", h_result[i]);
  }
  printf("%d\n\n", h_result[h_result.size() - 1]);

  return 0;
}
