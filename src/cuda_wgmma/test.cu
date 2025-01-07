#include <cstdio>
#include <bitset>

#include <cuda.h>
#include <cuda_fp16.h>

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

void gen_idx(int thread_idx) {
  int base_row0 = thread_idx / 4;
  int base_col  = thread_idx % 4;
  
  int idx00 = base_row0 * 32 + (base_col + 0 * 4) * 1;
  int idx01 = base_row0 * 32 + (base_col + 1 * 4) * 1;
  int idx02 = base_row0 * 32 + (base_col + 2 * 4) * 1;
  int idx03 = base_row0 * 32 + (base_col + 3 * 4) * 1;
  int idx04 = base_row0 * 32 + (base_col + 4 * 4) * 1;
  int idx05 = base_row0 * 32 + (base_col + 5 * 4) * 1;
  int idx06 = base_row0 * 32 + (base_col + 6 * 4) * 1;
  int idx07 = base_row0 * 32 + (base_col + 7 * 4) * 1;

  int base_row1 = base_row0 + 8;
  int idx08 = base_row1 * 32 + (base_col + 0 * 4) * 1;
  int idx09 = base_row1 * 32 + (base_col + 1 * 4) * 1;
  int idx10 = base_row1 * 32 + (base_col + 2 * 4) * 1;
  int idx11 = base_row1 * 32 + (base_col + 3 * 4) * 1;
  int idx12 = base_row1 * 32 + (base_col + 4 * 4) * 1;
  int idx13 = base_row1 * 32 + (base_col + 5 * 4) * 1;
  int idx14 = base_row1 * 32 + (base_col + 6 * 4) * 1;
  int idx15 = base_row1 * 32 + (base_col + 7 * 4) * 1;

  printf("idx00: %4d\n", idx00);
  printf("idx01: %4d\n", idx01);
  printf("idx02: %4d\n", idx02);
  printf("idx03: %4d\n", idx03);
  printf("idx04: %4d\n", idx04);
  printf("idx05: %4d\n", idx05);
  printf("idx06: %4d\n", idx06);
  printf("idx07: %4d\n", idx07);
  printf("idx08: %4d\n", idx08);
  printf("idx09: %4d\n", idx09);
  printf("idx10: %4d\n", idx10);
  printf("idx11: %4d\n", idx11);
  printf("idx12: %4d\n", idx12);
  printf("idx13: %4d\n", idx13);
  printf("idx14: %4d\n", idx14);
  printf("idx15: %4d\n", idx15);
}

void gen_idx_v2(int thread_idx) {
  int base_row0 = (thread_idx / 4) + (thread_idx / 32) * 8;
  int base_col  = (thread_idx % 4) * 2;
  
  int idx00 = base_row0 * 64 + (base_col + 0 * 8) * 1;
  int idx01 = idx00 + 1;

  int idx02 = base_row0 * 64 + (base_col + 1 * 8) * 1;
  int idx03 = idx02 + 1;

  int idx04 = base_row0 * 64 + (base_col + 2 * 8) * 1;
  int idx05 = idx04 + 1;

  int idx06 = base_row0 * 64 + (base_col + 3 * 8) * 1;
  int idx07 = idx06 + 1;

  int idx08 = base_row0 * 64 + (base_col + 4 * 8) * 1;
  int idx09 = idx08 + 1;

  int idx10 = base_row0 * 64 + (base_col + 5 * 8) * 1;
  int idx11 = idx10 + 1;

  int idx12 = base_row0 * 64 + (base_col + 6 * 8) * 1;
  int idx13 = idx12 + 1;

  int idx14 = base_row0 * 64 + (base_col + 7 * 8) * 1;
  int idx15 = idx14 + 1;

  int base_row1 = base_row0 + 8;
  int idx16 = base_row1 * 64 + (base_col + 0 * 8) * 1;
  int idx17 = idx16 + 1;

  int idx18 = base_row1 * 64 + (base_col + 1 * 8) * 1;
  int idx19 = idx18 + 1;

  int idx20 = base_row1 * 64 + (base_col + 2 * 8) * 1;
  int idx21 = idx20 + 1;

  int idx22 = base_row1 * 64 + (base_col + 3 * 8) * 1;
  int idx23 = idx22 + 1;

  int idx24 = base_row1 * 64 + (base_col + 4 * 8) * 1;
  int idx25 = idx24 + 1;

  int idx26 = base_row1 * 64 + (base_col + 5 * 8) * 1;
  int idx27 = idx26 + 1;

  int idx28 = base_row1 * 64 + (base_col + 6 * 8) * 1;
  int idx29 = idx28 + 1;

  int idx30 = base_row1 * 64 + (base_col + 7 * 8) * 1;
  int idx31 = idx30 + 1;

  printf("i = 0:");
  printf(" %4d", idx00); printf(" %4d", idx01); printf(" %4d", idx02); printf(" %4d", idx03); 
  printf(" %4d", idx04); printf(" %4d", idx05); printf(" %4d", idx06); printf(" %4d", idx07);
  printf(" %4d", idx08); printf(" %4d", idx09); printf(" %4d", idx10); printf(" %4d", idx11);
  printf(" %4d", idx12); printf(" %4d", idx13); printf(" %4d", idx14); printf(" %4d", idx15);
  printf("\n");

  printf("i = 1:");
  printf(" %4d", idx16); printf(" %4d", idx17); printf(" %4d", idx18); printf(" %4d", idx19); 
  printf(" %4d", idx20); printf(" %4d", idx21); printf(" %4d", idx22); printf(" %4d", idx23);
  printf(" %4d", idx24); printf(" %4d", idx25); printf(" %4d", idx26); printf(" %4d", idx27);
  printf(" %4d", idx28); printf(" %4d", idx29); printf(" %4d", idx30); printf(" %4d", idx31);
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
  for (int i = 0; i < 128; i++) {
    printf("========================\n");
    printf("thread_idx = %4d\n", i);
    printf("========================\n");
    // gen_idx(i);
    gen_idx_v2(i);
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
