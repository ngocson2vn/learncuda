
#include <cuda_runtime.h>

void launch_kernel(float* data, int n);

int const n = 3;

int main() {
  // Create host data
  float* h_data = new float[n];
  for (int i = 0; i < n; i++) {
    h_data[i] = 1.0 * i;
  }

  // Transfer data from host to device
  float* d_data;
  cudaMalloc((void**)&d_data, n * sizeof(float));
  cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);

  launch_kernel(d_data, n);
  cudaDeviceSynchronize();
  return 0;
}
