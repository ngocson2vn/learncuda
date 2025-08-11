#include <stdio.h>
#include <cuda_runtime.h>

__global__ void printf_kernel(float* a) {
  int idx = threadIdx.x;
  printf("threadIdx.x = %d, pointer address: %p\n", idx, (a + idx));
  printf("threadIdx.x = %d, a[%d] = %f\n", idx, idx, *(a + idx));
  *(a + idx) *= 2;
}

int main(int argc, char** argv) {
  // Transfer data from host to device
  int N = 10;
  float* a = new float[N];
  for (int i = 0; i < N; i++) {
    *(a + i) = 1.0 * i;
  }

  float* d_a;
  // cudaMalloc((void**)&d_a, N * sizeof(float));
  // cudaMemcpyAsync(d_a, a, N * sizeof(float), cudaMemcpyHostToDevice);

  printf_kernel<<<1, 1>>>(d_a);
  cudaError_t err = cudaDeviceSynchronize(); 
  if (err != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(err)); }
}
