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


template <int SMEM_SIZE = 1024>
__global__ void dumb_kernel(float* data)
{
  assert(SMEM_SIZE >= blockDim.x && "SMEM_SIZE is less than block size");
  assert(SMEM_SIZE % blockDim.x == 0 && "SMEM_SIZE is not divisible by block size");

  // if (threadIdx.x == 0) printf("SMEM_SIZE: %d\n", SMEM_SIZE);
  __shared__ float smem_data[SMEM_SIZE]; // for each block of threads

  // Compute partition size that each thread is responsible for
  int partition_size = SMEM_SIZE / blockDim.x;
  // if (threadIdx.x == 0) printf("partition_size: %d\n", partition_size);

  // Compute data offset for each thread
  int smem_offset = threadIdx.x * partition_size;
  int gmem_offset = blockIdx.x * SMEM_SIZE + smem_offset;
  
  // Copy a partition of gmem to smem
  for (int i = 0; i < partition_size; i++) {
    smem_data[smem_offset + i] = data[gmem_offset + i];
  }
  __syncthreads();

  // Read and write smem
  float sum = 0;
  for (int k = 0; k < SMEM_SIZE; k++) {
    sum += smem_data[k];
  }

  for (int i = 0; i < partition_size; i++) {
    smem_data[smem_offset + i] = smem_data[smem_offset + i] / sum;

    // Copy smem back to gmem
    data[gmem_offset + i] = smem_data[smem_offset + i];
  }
}

int main(int argc, char** argv) {
  constexpr int kDataSize = 1024 * 1024;
  constexpr int SMEM_SIZE = 128 * 128;

  static_assert(kDataSize % SMEM_SIZE == 0, "data size is not divisible by SMEM_SIZE");
  dim3 grid_dim(kDataSize / SMEM_SIZE);

  static_assert(SMEM_SIZE % 32 == 0, "SMEM_SIZE is not divisible by warp size");
  dim3 block_dim(32);

  // Create host vector
  thrust::host_vector<float> h_data(kDataSize);
  for (int i = 0; i < kDataSize; i++) h_data[i] = 1.0 * i;

  device_init(0);

  // Copy host vector to gmem
  thrust::device_vector<float> d_data = h_data;

  // Launch kernel
  void* kernel = reinterpret_cast<void*>(dumb_kernel<SMEM_SIZE>);
  float* gmem_ptr = d_data.data().get();
  void* kernel_args[] = {static_cast<void*>(&gmem_ptr)};
  CUDA_CHECK_ERROR(cudaLaunchKernel(kernel, grid_dim, block_dim, kernel_args));

  thrust::host_vector<float> h_result = d_data;
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());

  return 0;
}
