#include "cutlass/util/command_line.h"

#include "scale_tma_kernel.h"
#include "tma_copy.h"
#include "tma_copy_multicast.h"

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

int main(int argc, char const **argv) {

  cutlass::CommandLine cmd(argc, argv);
  // Parses the command line

  int M, N, iterations;  
  cmd.get_cmd_line_argument("M", M, 16384);
  cmd.get_cmd_line_argument("N", N, 16384);
  cmd.get_cmd_line_argument("iterations", iterations, 1);

  std::cout << "(M, N): " << M << ", " << N << std::endl;

  device_init(0);
  copy_host_tma_load_and_store_kernel(M, N, iterations);

  // scaleTmaKernelHost(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<true, 2>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<false, 2>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<true, 4>(M, N, iterations);
  // copy_host_tma_load_and_store_kernel_multicast<false, 4>(M, N, iterations);

  return 0;
}
