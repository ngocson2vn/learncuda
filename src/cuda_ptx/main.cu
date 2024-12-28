#include <cstdio>
#include <cstdint>
#include <device_launch_parameters.h>

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

__global__ void test_kernel() {
  uint32_t is_elected_lane = elect_one_sync();
  printf("threadIdx.x = %d is_elected_lane = %d\n", threadIdx.x, is_elected_lane);
}

int main(int argc, char** argv) {
  // Init cuda
  device_init(0);

  dim3 blocks(1);
  dim3 threads_per_block(32);
  test_kernel<<<blocks, threads_per_block>>>();
  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}
