#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cstdio>

#include "ws.h"

using barrier = cuda::barrier<cuda::thread_scope_block>;

__device__ void producer(barrier& ready, barrier& filled, float* buffer, float* in, int N, int buffer_len) {
  ready.arrive_and_wait(); /* wait for buffer to be ready to be filled */

  /* produce, i.e., fill in, buffer  */
  int in_idx = blockIdx.x * (blockDim.x / 2) + threadIdx.x;
  // printf("in_idx: %d\n", in_idx);
  buffer[threadIdx.x] = in[in_idx];

  filled.arrive(); /* buffer is filled */
}

__device__ void consumer(barrier& ready, barrier& filled, float* buffer, float* out, int N, int buffer_len) {
  ready.arrive(); /* buffer is ready for initial fill */
  filled.arrive_and_wait(); /* wait for buffer to be filled */
  
  /* consume buffer */
  int out_idx = blockIdx.x * (blockDim.x / 2) + threadIdx.x - kWarpSize;
  out[out_idx] = 2 * buffer[threadIdx.x - kWarpSize];
}

// N is the total number of float elements in arrays in and out
__global__ void producer_consumer_pattern(int N, int buffer_len, float* in, float* out) {

  // Shared memory buffer declared below is of size buffer_len
  __shared__ extern float buffer[];

  __shared__ barrier bar[2];


  auto block = cooperative_groups::this_thread_block();
  if (block.thread_rank() == 0) {
    init(bar, block.size());
    init(bar+1, block.size());
  }
  block.sync();

  if (block.thread_rank() < kWarpSize) {
    producer(bar[0], bar[1], buffer, in, N, buffer_len);
  } else {
    consumer(bar[0], bar[1], buffer, out, N, buffer_len);
  }
}

void process(float* h_in, float* h_out, int N) {
  // Create a stream
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  // Transfer data from host to device
  float* d_in;
  cudaMalloc((void**)&d_in, N * sizeof(float));
  cudaMemcpyAsync(d_in, h_in, N * sizeof(float), cudaMemcpyHostToDevice, stream);

  float* d_out;
  cudaMalloc((void**)&d_out, N * sizeof(float));

  // Process data
  dim3 numBlocks(N / kWarpSize, 1, 1);
  dim3 numThreads(kWarpSize * 2, 1, 1); // 1 producer warp, 1 consumer warp
  int buffer_len = kWarpSize;
  int smem_size = buffer_len * sizeof(float);
  cudaFuncSetAttribute(producer_consumer_pattern, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size);
  producer_consumer_pattern<<<numBlocks, numThreads, smem_size, stream>>>(N, buffer_len, d_in, d_out);

  // Transfer data from device to host
  cudaMemcpyAsync(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost, stream);

  // Sync GPU
  cudaDeviceSynchronize();

  // Free device memory
  cudaFree(d_in);
  cudaFree(d_out);
}
