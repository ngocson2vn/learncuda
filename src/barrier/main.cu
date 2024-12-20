#include <cstdio>

#include <cuda/barrier>
#include <cooperative_groups.h>
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

__device__ void compute(float* data, int size) {
  int idx = threadIdx.x;
  data[idx] = data[idx] * 2;
}

__global__ void split_arrive_wait(float *data, int size) {
  using barrier = cuda::barrier<cuda::thread_scope_block>;
  __shared__  barrier bar;
  auto block = cooperative_groups::this_thread_block();

  if (block.thread_rank() == 0) {
      init(&bar, block.size()); // Initialize the barrier with expected arrival count
  }
  block.sync();

  /* code before arrive */
  compute(data, size);

  /* this thread arrives. Arrival does not block a thread */
  barrier::arrival_token token = bar.arrive();

  /* wait for all threads participating in the barrier to complete bar.arrive() */
  bar.wait(std::move(token));

  /* code after wait */
  if (block.thread_rank() == 0) {
    printf("thread %d: ", threadIdx.x);
    for (int i = 0; i < size - 1; i++) {
      printf("%.2f ", data[i]);
    }
    printf("%.2f", data[size - 1]);
    printf("\n\n");
  }
}

int main(int argc, char** argv) {
  constexpr int kThreads = 32;
  thrust::host_vector<float> h_data(kThreads);
  for (int i = 0; i < kThreads; i++) h_data[i] = i;

  device_init(0);

  thrust::device_vector<float> d_data = h_data;
  dim3 grid_dim(1);
  dim3 block_dim(kThreads);
  split_arrive_wait<<<grid_dim, block_dim>>>(d_data.data().get(), kThreads);
}
