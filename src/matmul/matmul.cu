#include "matmul.h"

//====================================================================================================================================
// Naive matmul
//====================================================================================================================================

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
  int width = dimsA.x;
  int P = dimsA.y;
  int Q = dimsB.x;
  naive_matrix_multiply<T><<<numBlocks, threadsPerBlock, 0, stream>>>(d_MatA, d_MatB, d_MatC, width, P, Q);

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
template int* matmul<int, 4>(int* MatA, int* MatB, const dim3& dimsA, const dim3& dimsB);
template int* matmul<int, 32>(int* MatA, int* MatB, const dim3& dimsA, const dim3& dimsB);

template float* matmul<float, 2>(float* MatA, float* MatB, const dim3& dimsA, const dim3& dimsB);
template float* matmul<float, 4>(float* MatA, float* MatB, const dim3& dimsA, const dim3& dimsB);
template float* matmul<float, 32>(float* MatA, float* MatB, const dim3& dimsA, const dim3& dimsB);


//====================================================================================================================================
// Shared memory matmul
//====================================================================================================================================
template <typename T, int BLOCK_SIZE>
__global__ void smem_matrix_multiply(const T* A, const T* B, T* C, const int widthA, const int widthB) {
  //=============================================
  // Prologue
  //=============================================
  const int block_row = blockIdx.y;
  const int block_col = blockIdx.x;
  const int kBlocks = widthA / BLOCK_SIZE;
  T Cval = 0;

  //=============================================
  // Mainloop
  //=============================================
  // Each thread performs kBlocks operations
  for (int m = 0; m < kBlocks; m++) {
    // Define sub-matrices of A and B
    __shared__ T Asub[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ T Bsub[BLOCK_SIZE * BLOCK_SIZE];

    // Load sub-matrices of A and B onto shared memory
    // Each thread loads only one element
    int hA = block_row * BLOCK_SIZE + threadIdx.y;
    Asub[threadIdx.y * BLOCK_SIZE + threadIdx.x] = A[hA * widthA + m * BLOCK_SIZE + threadIdx.x];

    int hB = m * BLOCK_SIZE + threadIdx.y;
    Bsub[threadIdx.y * BLOCK_SIZE + threadIdx.x] = B[hB * widthB + block_col * BLOCK_SIZE + threadIdx.x];

    // Make sure that sub-matrices are loaded
    __syncthreads();

    // Matmul
    for (int k = 0; k < BLOCK_SIZE; k++) {
      Cval += Asub[threadIdx.y * BLOCK_SIZE + k] * Bsub[k * BLOCK_SIZE + threadIdx.x];
    }

    // Make sure that all threads have finished matmul
    // before proceeding to next round
    __syncthreads();
  } // Mainloop

  //=============================================
  // Epilogue
  //=============================================
  // Store the result into C matrix
  const int row = block_row * blockDim.y + threadIdx.y;
  const int col = block_col * blockDim.x + threadIdx.x;
  C[row * widthB + col] = Cval;
}

template <typename T, int BLOCK_SIZE>
T* smem_matmul(T* MatA, T* MatB, const dim3& dimsA, const dim3& dimsB) {
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
  int widthA = dimsA.x;
  int widthB = dimsB.x;
  smem_matrix_multiply<T, BLOCK_SIZE><<<numBlocks, threadsPerBlock, 0, stream>>>(d_MatA, d_MatB, d_MatC, widthA, widthB);

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

template int* smem_matmul<int, 2>(int* MatA, int* MatB, const dim3& dimsA, const dim3& dimsB);
template int* smem_matmul<int, 4>(int* MatA, int* MatB, const dim3& dimsA, const dim3& dimsB);
template int* smem_matmul<int, 32>(int* MatA, int* MatB, const dim3& dimsA, const dim3& dimsB);

template float* smem_matmul<float, 2>(float* MatA, float* MatB, const dim3& dimsA, const dim3& dimsB);
template float* smem_matmul<float, 4>(float* MatA, float* MatB, const dim3& dimsA, const dim3& dimsB);
template float* smem_matmul<float, 32>(float* MatA, float* MatB, const dim3& dimsA, const dim3& dimsB);
