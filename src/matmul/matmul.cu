#include "matmul.h"

template <typename T>
__global__ void naive_matrix_multiply(const T* A, const T* B, T* C, int width, int P, int Q)
{
  int r = blockIdx.y * blockDim.y + threadIdx.y;   
  int c = blockIdx.x * blockDim.x + threadIdx.x;
  // check boundry conditions
  if (r < P && c < Q){
    // do the multiplication for one row and col
    T value = 0;
    for(int k = 0; k < width; k++){
      value += A[r * width + k] * B[k * Q + c];
    }

    // store the result
    C[r * Q + c] = value;
  }
}

template <typename T, int BLOCK_SIZE>
T* matmul(T* MatA, T* MatB, const dim3& dimsA, const dim3& dimsB) {
  // Create a stream
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Transfer data from host to device
  int sizeA = dimsA.x * dimsA.y;
  T* d_MatA;
  cudaMalloc((void**)&d_MatA, sizeA * sizeof(T));
  cudaMemcpyAsync(d_MatA, MatA, sizeA * sizeof(T), cudaMemcpyHostToDevice, stream);

  int sizeB = dimsB.x * dimsB.y;
  T* d_MatB;
  cudaMalloc((void**)&d_MatB, sizeB * sizeof(T));
  cudaMemcpyAsync(d_MatB, MatB, sizeB * sizeof(T), cudaMemcpyHostToDevice, stream);

  int sizeC = dimsA.y * dimsB.x;
  T* d_MatC;
  cudaMalloc((void**)&d_MatC, sizeC * sizeof(T));

  // Launch matmul kernel
  dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
  dim3 numBlocks(
    (dimsB.x + threadsPerBlock.x - 1) / threadsPerBlock.x,
    (dimsA.y + threadsPerBlock.y - 1) / threadsPerBlock.y
  );
  naive_matrix_multiply<T><<<numBlocks, threadsPerBlock, 0, stream>>>(d_MatA, d_MatB, d_MatC, dimsA.x, dimsA.y, dimsB.x);

  // Transfer results from device to host
  T* MatC = new T[sizeC];
  cudaMemcpyAsync(MatC, d_MatC, sizeC * sizeof(T), cudaMemcpyDeviceToHost, stream);

  cudaDeviceSynchronize();

  // Free device memory
  cudaFree(d_MatA);
  cudaFree(d_MatB);
  cudaFree(d_MatC);

  return MatC;
}

template int* matmul<int, 2>(int* MatA, int* MatB, const dim3& dimsA, const dim3& dimsB);

template int* matmul<int, 32>(int* MatA, int* MatB, const dim3& dimsA, const dim3& dimsB);

template float* matmul<float, 2>(float* MatA, float* MatB, const dim3& dimsA, const dim3& dimsB);

template float* matmul<float, 32>(float* MatA, float* MatB, const dim3& dimsA, const dim3& dimsB);