#include <cstdio>
#include <cstdlib>

#include "cuda.h"

void print_mat(int* a, const char* name, const int N) {
  printf("%s\n", name);
  printf("==============================================================\n");
  for (size_t i = 0; i < N; i++) {
    printf("i = %-3d:", i);
    for (size_t j = 0; j < N; j++) {
      printf(" %-6d", a[N * i + j]);
      if (j == 3 && j < N) {
        printf(" ... %-6d", a[N * i + (N - 1)]);
        break;
      }
    }
    printf("\n");
    if (i == 3 && i < N) {
      i = N - 1;
      printf(".\n.\n.\n");
      printf("i = %-3d:", i);
      for (size_t j = 0; j < N; j++) {
        printf(" %-6d", a[N * i + j]);
        if (j == 3 && j < N) {
          printf(" ... %-6d", a[N * i + (N - 1)]);
          break;
        }
      }
      break;
    }
  }
}

int main() {
  const int N = 1e3;
  int* MatA = (int*)malloc(N * N * sizeof(int));
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      MatA[N * i + j] = i + j + 1;
    }
  }

  int* MatB = (int*)malloc(N * N * sizeof(int));
  for (size_t i = 0; i < N; i++) {
    for (size_t j = 0; j < N; j++) {
      MatB[N * i + j] = i + j + 1;
    }
  }

  print_mat(MatA, "MatA", N);
  printf("\n\n");
  print_mat(MatB, "MatB", N);
  printf("\n\n");

  // Matrix addition
  int* MatC = mat_add(MatA, MatB, N);
  print_mat(MatC, "MatC", N);
  printf("\n");

  free(MatA);
  free(MatB);
  free(MatC);
}