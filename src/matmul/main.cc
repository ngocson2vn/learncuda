#include <cstdio>
#include <cstdlib>
#include <random>

#include "matmul.h"

void print_mat(int* a, const char* name, const dim3& dims) {
  printf("%s\n", name);
  printf("==============================================================\n");
  for (size_t j = 0; j < dims.y; j++) {
    printf("j = %-3d:", j);
    for (size_t i = 0; i < dims.x; i++) {
      printf(" %-6d", a[j * dims.x + i]);
      if (i == 10 && i < dims.x) {
        printf(" ... %-6d", a[j * dims.x + (dims.x - 1)]);
        break;
      }
    }
    printf("\n");
    if (j == 10 && j < dims.y) {
      j = dims.y - 1;
      printf(".\n.\n.\n");
      printf("j = %-3d:", j);
      for (size_t i = 0; i < dims.x; i++) {
        printf(" %-6d", a[j * dims.x + i]);
        if (i == 10 && i < dims.x) {
          printf(" ... %-6d", a[j * dims.x + (dims.x - 1)]);
          break;
        }
      }
      break;
    }
  }
}

int* matmul_cpu(int* MatA, int* MatB, const dim3& dimsA, const dim3& dimsB) {
  int* MatC = new int[dimsA.y * dimsB.x];
  for (int j = 0; j < dimsA.y; j++) {
    for (int i = 0; i < dimsB.x; i++) {
      for (int k = 0; k < dimsA.x; k++) {
        MatC[j * dimsB.x + i] += MatA[j * dimsA.x + k] * MatB[k * dimsB.x + i];
      }
    }
  }

  return MatC;
}

int main() {
  const int kBlockSize = 2;
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist3(1, 3); // distribution in range [1, 3]

  const dim3 dimsA(3 * kBlockSize, 2 * kBlockSize); // rows(y): 2 x columns(x): 3
  int* MatA = new int[dimsA.x * dimsA.y];
  for (size_t i = 0; i < dimsA.y; i++) {
    for (size_t j = 0; j < dimsA.x; j++) {
      MatA[i * dimsA.x + j] = dist3(rng);
    }
  }

  const dim3 dimsB(2 * kBlockSize, 3 * kBlockSize); // rows(y): 3 x columns(x): 4
  int* MatB = new int[dimsB.x * dimsB.y];
  for (size_t i = 0; i < dimsB.y; i++) {
    for (size_t j = 0; j < dimsB.x; j++) {
      MatB[i * dimsB.x + j] = dist3(rng);
    }
  }

  print_mat(MatA, "MatA", dimsA);
  printf("\n\n");
  print_mat(MatB, "MatB", dimsB);
  printf("\n\n");

  const dim3 dimsC(dimsB.x, dimsA.y);

  // CPU
  int* MatC_cpu = matmul_cpu(MatA, MatB, dimsA, dimsB);
  print_mat(MatC_cpu, "MatC_cpu", dimsC);
  printf("\n");

  // GPU
  int* MatC_gpu = matmul<int, kBlockSize>(MatA, MatB, dimsA, dimsB);
  print_mat(MatC_gpu, "MatC_gpu", dimsC);
  printf("\n");

  free(MatA);
  free(MatB);
  free(MatC_cpu);
  free(MatC_gpu);
}
