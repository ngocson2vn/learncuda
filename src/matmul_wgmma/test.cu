#include <cstdio>
#include <bitset>

#include <cuda.h>
#include <cuda_fp16.h>

#include "fprint_mat.h"

#define CUDA_CHECK_ERROR(e)                                    \
do {                                                           \
  cudaError_t code = (e);                                      \
  if (code != cudaSuccess) {                                   \
    fprintf(stderr, "<%s:%4d> %s:\n    %s: %s\n",               \
            __FILE__, __LINE__, #e,                            \
            cudaGetErrorName(code), cudaGetErrorString(code)); \
    fflush(stderr);                                            \
    exit(1);                                                   \
  }                                                            \
} while (0)

void gen_indexes(std::size_t* indexes_tile_D, std::size_t* shape_tile_D, std::size_t* stride_tile_D, std::size_t thread_idx) {
  int baseRow = (thread_idx / 4) + (thread_idx / 32) * 8;
  int baseCol  = (thread_idx % 4) * 2;
  int k = 0;
  int idx = 0;
  for (int i = 0; i < (shape_tile_D[0] / 32); i++) {
    int fragment_idx = k;
    for (int j = 0; j < (shape_tile_D[1] / 8); j++) {
      idx = (baseRow + i * 8) * stride_tile_D[0] + (baseCol + j * 8) * stride_tile_D[1];
      indexes_tile_D[fragment_idx]   = idx;
      indexes_tile_D[fragment_idx+1] = idx + 1;
      fragment_idx += 4;
    }

    k += 2;
  }
}

template<int M = 64, int K = 16>
__global__ void test_kernel() {
  __shared__ alignas(128) half sA[M * K];

  auto mde = [](uint64_t x) -> uint64_t { return (x & 0x3FFFF) >> 0x4; };

  if (threadIdx.x == 0) {
    uint64_t a0 = reinterpret_cast<uint64_t>(sA);
    printf("sA0: %x\n", a0);

    uint64_t a1 = static_cast<uint32_t>(__cvta_generic_to_shared(sA));
    printf("sA1: %x\n", a1);

    uint64_t a2 = mde(reinterpret_cast<uint64_t>(sA));
    printf("sA2: %x\n", a2);
  }
}

int main(int argc, char** argv) {
  constexpr std::size_t M_TILE = 64;
  constexpr std::size_t N_TILE = 64;
  // std::size_t indexes_tile_D[M_TILE * N_TILE];
  // for (std::size_t i = 0; i < M_TILE*N_TILE; i++) {
  //   indexes_tile_D[i] = 0;
  // }

  // std::size_t shape_tile_D[] = {M_TILE, N_TILE};
  // std::size_t stride_tile_D[] = {N_TILE, 1};
  // std::string outputFile = "indexes_tile_D";
  // for (int tid = 0; tid < 128; tid++) {
  //   printf("========================\n");
  //   printf("thread_idx = %4d\n", tid);
  //   printf("========================\n");
  //   gen_indexes(indexes_tile_D, shape_tile_D, stride_tile_D, tid);
  //   std::string tmpOutput = outputFile + "_tid_" + std::to_string(tid) + ".txt";
  //   FILE* file_ptr = get_file_ptr(tmpOutput.c_str());
  //   fprint_mat(file_ptr, "indexes_tile_D", indexes_tile_D, shape_tile_D, stride_tile_D);
  //   printf("\n\n");
  // }

  int fragment_size = 32;
  std::size_t indexes_tile_D[fragment_size];
  std::size_t shape_tile_D[] = {M_TILE, N_TILE};
  std::size_t stride_tile_D[] = {N_TILE, 1};

  const char* eformat = " %10d";

  for (int tid = 0; tid < 128; tid++) {
    printf("========================\n");
    printf("thread_idx = %4d\n", tid);
    printf("========================\n");

    for (std::size_t i = 0; i < fragment_size; i++) {
      indexes_tile_D[i] = 0;
    }

    gen_indexes(indexes_tile_D, shape_tile_D, stride_tile_D, tid);

    int fragment_idx = 0;
    for (int i = 0; i < 2; i++) {
      fragment_idx = i * 2;
      for (int j = 0; j < 8; j++) {
        printf(eformat, indexes_tile_D[fragment_idx]);
        printf(eformat, indexes_tile_D[fragment_idx + 1]);
        fragment_idx += 4;
      }
      printf("\n");
    }

    printf("\n\n");
  }

  // uint64_t d1 = 0x400;
  // printf("%x: %s\n", d1, std::bitset<64>(d1).to_string().c_str());
  // uint64_t d2 = 0xafffcd0;
  // printf("%x: %s\n", d2, std::bitset<64>(d2).to_string().c_str());

  // {
  //   cudaLaunchConfig_t config = {0};
  //   // The grid dimension is not affected by cluster launch, and is still enumerated
  //   // using number of blocks.
  //   // The grid dimension should be a multiple of cluster size.
  //   config.gridDim = dim3(1);
  //   config.blockDim = dim3(128);
  //   // config.stream = stream;

  //   cudaLaunchAttribute attribute[1];
  //   attribute[0].id = cudaLaunchAttributeClusterDimension;
  //   attribute[0].val.clusterDim.x = 1; // Cluster size in X-dimension
  //   attribute[0].val.clusterDim.y = 1;
  //   attribute[0].val.clusterDim.z = 1;
  //   config.attrs = attribute;
  //   config.numAttrs = 1;

  //   CUDA_CHECK_ERROR(
  //     cudaLaunchKernelEx(
  //       &config,
  //       test_kernel
  //     )
  //   );
  // }

  // CUDA_CHECK_ERROR(cudaDeviceSynchronize());
}
