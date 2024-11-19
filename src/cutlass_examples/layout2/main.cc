#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>
#include <string>

#include "gemm.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/helper_cuda.hpp"

constexpr static int kMaxElement = 100;

template <typename T>
void print_mat(const char* name, T* a, const dim3& dims) {
  // Header
  printf("%s\n", name);
  int num_chars = 8 + 7 * dims.y;
  printf("%s\n", std::string(num_chars, '-').c_str());
  printf("      j:");
  for (size_t j = 0; j < dims.y; j++) {
    printf(" %6d", j);
  }
  printf("\n%s\n", std::string(num_chars, '-').c_str());

  // Body
  for (size_t i = 0; i < dims.x; i++) {
    printf("i = %-3d:", i);
    for (size_t j = 0; j < dims.y; j++) {
      printf(" %6.1f", a[i + j * dims.x]);
      if (j == kMaxElement && j < dims.y) {
        printf(" ... %6.1f", a[i + (dims.y - 1) * dims.x]);
        break;
      }
    }
    printf("\n");
    if (i == kMaxElement && i < dims.x) {
      i = dims.x - 1;
      printf(".\n.\n.\n");
      printf("i = %-3d:", i);
      for (size_t j = 0; j < dims.y; j++) {
        printf(" %6.1f", a[i + j * dims.x]);
        if (j == kMaxElement && j < dims.y) {
          printf(" ... %6.1f", a[i + (dims.y - 1) * dims.x]);
          break;
        }
      }
      break;
    }
  }
}

template <typename T>
T* matmul_nt_cpu(T* MatA, const int sizeA, const int ldA, T* MatB, const int sizeB, const int ldB) {
  assert((sizeA / ldA) == (sizeB / ldB) && "K mismatch");
  const int bigK = sizeA / ldA;
  T* MatC = new T[ldA * ldB];
  for (int i = 0; i < ldA; i++) {
    for (int j = 0; j < ldB; j++) {
      for (int k = 0; k < bigK; k++)
      MatC[i + j * ldA] += MatA[i + k * ldA] * MatB[j + k * ldB];
    }
  }

  return MatC;
}

template <typename SALayout, typename TCLayout>
__global__ void partition_kernel(SALayout sA, TCLayout tC) {
  // Partition sA (M,K) by the rows of tC
  cute::Tensor tCsA = cute::local_partition(sA, tC, threadIdx.x, cute::Step<cute::_1, cute::X>{});   // (THR_M,BLK_K)

  cute::print("thread%d\n", threadIdx);
  cute::print("  sA : "); cute::print(  sA); cute::print("\n");
  cute::print("tCsA : "); cute::print(tCsA); cute::print("\n");
}

int main(int argc, char** argv) {
  int M = 16;
  if (argc >= 2)
    sscanf(argv[1], "%d", &M);

  int N = 16;
  if (argc >= 3)
    sscanf(argv[2], "%d", &N);

  int k = 8;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  int bM = 8;
  int bN = 8;
  int bK = 4;
  auto sA = cute::make_layout(cute::make_shape(bM, bK));                 // (M,k) -> smem_idx; M-major

  dim3 threadsPerBlock(16);
  dim3 dimGrid(cute::size(cute::ceil_div(M, bM)),
               cute::size(cute::ceil_div(N, bN)));
  partition_kernel<<<dimGrid, threadsPerBlock, 0, stream>>>

  return 0;
}
