#include <cstdio>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <device_launch_parameters.h>

#include "fprint_mat.h"

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

__global__ void tensor_core_example_8x8x16(
  int32_t       *D,
  int32_t const *A,
  int32_t const *B,
  int32_t const *C) {

  // Compute the coordinates of accesses to A and B matrices
  int outer = threadIdx.x / 4; // m or n dimension
  int inner = threadIdx.x % 4; // k dimension

  // Compute the coordinates for the accumulator matrices
  int c_row = threadIdx.x / 4;
  int c_col = 2 * (threadIdx.x % 4);

  // Compute linear offsets into each matrix
  int ab_idx = outer * 4 + inner;
  int cd_idx = c_row * 8 + c_col;

  printf("blockIdx.x: %d, threadIdx.x: %d, cd_idx: %d\n", blockIdx.x, threadIdx.x, cd_idx);

  // Issue Tensor Core operation
  asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0, %1}, {%2}, {%3}, {%4, %5};\n"
    : "=r"(D[cd_idx]), "=r"(D[cd_idx + 1])
    : "r"(A[ab_idx]), "r"(B[ab_idx]), "r"(C[cd_idx]), "r"(C[cd_idx + 1])
  );
}

int main(int argc, char** argv) {
  int m = 8;
  int n = 8;
  int k = 4;
  thrust::host_vector<int32_t> h_A(m*k);
  thrust::host_vector<int32_t> h_B(n*k);
  thrust::host_vector<int32_t> h_C(m*n);
  thrust::host_vector<int32_t> h_D(m*n);

  // Initialize the tensors
  int32_t val = 1;
  int32_t mask = 1;
  for (int i = 0; i < 3; i++) {
    mask = mask << 8;
    val = val | mask;
  }

  for (int j = 0; j < m*k; ++j) h_A[j] = int32_t(val);
  for (int j = 0; j < n*k; ++j) h_B[j] = int32_t(val);
  for (int j = 0; j < m*n; ++j) h_C[j] = int32_t(0);
  for (int j = 0; j < m*n; ++j) h_D[j] = int32_t(0);

  fprint_mat(stdout, "h_A", h_A.data(), dim3(m, k, 1));
  fprintf(stdout, "\n\n");

  fprint_mat(stdout, "h_B", h_B.data(), dim3(k, n, 1));
  fprintf(stdout, "\n\n");

  // Init cuda
  device_init(0);

  thrust::device_vector<int32_t> d_A = h_A;
  thrust::device_vector<int32_t> d_B = h_B;
  thrust::device_vector<int32_t> d_C = h_C;
  thrust::device_vector<int32_t> d_D = h_D;

  dim3 blocks(1);
  dim3 threads_per_block(32);
  tensor_core_example_8x8x16<<<blocks, threads_per_block>>>(d_D.data().get(), d_A.data().get(), d_B.data().get(), d_C.data().get());
  
  // Copy output matrix
  h_D = d_D;

  fprintf(stdout, "\n\n");
  fprint_mat(stdout, "h_D", h_D.data(), dim3(m, n, 1));

  CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}
