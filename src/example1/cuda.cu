#include "cuda.h"

// Kernel - Adding two matrices MatA and MatB
__global__ void cuda_mat_add(int* d_MatA, int* d_MatB, int* d_MatC, const int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i < N && j < N) d_MatC[N * i + j] = d_MatA[N * i + j] + d_MatB[N * i + j];
}

int* mat_add(int* MatA, int* MatB, const int N) {
  // Create a stream
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Transfer data from host to device
  int* d_MatA;
  cudaMalloc((void**)&d_MatA, N * N * sizeof(int));
  cudaMemcpyAsync(d_MatA, MatA, N * N * sizeof(int), cudaMemcpyHostToDevice, stream);

  int* d_MatB;
  cudaMalloc((void**)&d_MatB, N * N * sizeof(int));
  cudaMemcpyAsync(d_MatB, MatB, N * N * sizeof(int), cudaMemcpyHostToDevice, stream);

  int* d_MatC;
  cudaMalloc((void**)&d_MatC, N * N * sizeof(int));

  // Launch MatAdd kernel
  dim3 threadsPerBlock(16, 16);
  dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, (N + threadsPerBlock.y - 1) / threadsPerBlock.y);
  cuda_mat_add<<<numBlocks, threadsPerBlock, 0, stream>>>(d_MatA, d_MatB, d_MatC, N);

  // Transfer results from device to host
  int* MatC = (int*)calloc(N * N, sizeof(int));
  cudaMemcpyAsync(MatC, d_MatC, N * N * sizeof(int), cudaMemcpyDeviceToHost, stream);

  cudaDeviceSynchronize();

  // Free device memory
  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);

  return MatC;
}
