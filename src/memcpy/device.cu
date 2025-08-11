
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void testKernel(float* data, int n) {
  // Dummy kernel
  printf("testKernel with params size %d bytes\n", n);
}

void launch_kernel(float* data, int n) {
  testKernel<<<1, 1>>>(data, n);
}