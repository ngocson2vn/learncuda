#include <cstdio>
#include <cuda_runtime.h>


__global__ void test_kernel() {
  constexpr int N = 128;
  __shared__ bool done[N];

  // Initialize counter
  if (threadIdx.x == 0) {
    for (int i = 0; i < N; i++) {
      done[i] = false;
    }
  }

  __syncthreads();

  // Wait for threadIdx.x - 1 to complete
  if (threadIdx.x > 0) {
    while (!done[threadIdx.x - 1]) {
      __nanosleep(1000);
    }
  }

  done[threadIdx.x] = true;
  printf("threadIdx.x = %3d is done!\n", threadIdx.x);
}

int main(int argc, char** argv) {
  test_kernel<<<1, 32>>>();
  cudaError_t err = cudaDeviceSynchronize(); 
  if (err != cudaSuccess) { printf("Error: %s\n", cudaGetErrorString(err)); }
}
