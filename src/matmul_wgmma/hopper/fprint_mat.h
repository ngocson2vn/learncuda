#pragma once

#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <device_launch_parameters.h>

#include <cuda_bf16.h>

using bf16_t = nv_bfloat16;

FILE* get_file_ptr(const char* file_path) {
  FILE *fptr;
  fptr = fopen(file_path, "w");
  if(fptr == NULL) {
    printf("Failed to open %s\n", file_path);
    exit(1);
  }

  return fptr;
}

constexpr static int kMaxElement = 10000000;

template <typename T, int digits = 10, int mantissa = 3>
void fprint_mat(FILE* file_ptr, const char* name, const thrust::host_vector<T>& a, const char indices[2], const int shape[2], const int stride[2]) {
  std::string header_fmt = "%" + std::to_string(digits) + "d";
  std::string element_fmt = "%" + std::to_string(digits) + "." + std::to_string(mantissa) + "f";

  // Header
  fprintf(file_ptr, "%s\n", name);
  int num_chars = 8 + digits * shape[1];
  fprintf(file_ptr, "%s\n", std::string(num_chars, '-').c_str());
  fprintf(file_ptr, "      %c:", indices[1]);
  for (int j = 0; j < shape[1]; j++) {
    fprintf(file_ptr, header_fmt.c_str(), j);
  }
  fprintf(file_ptr, "\n%s\n", std::string(num_chars, '-').c_str());

  // Body
  for (int i = 0; i < shape[0]; i++) {
    fprintf(file_ptr, "%c = %3d:", indices[0], i);
    for (int j = 0; j < shape[1]; j++) {
      fprintf(file_ptr, element_fmt.c_str(), (float)a[i * stride[0] + j * stride[1]]);
      if (j == kMaxElement && j < shape[1]) {
        std::string efmt = " ... " + element_fmt;
        fprintf(file_ptr, efmt.c_str(), (float)a[i * stride[0] + (shape[1] - 1) * stride[1]]);
        break;
      }
    }
    fprintf(file_ptr, "\n");
    if (i == kMaxElement && i < shape[0]) {
      i = shape[0] - 1;
      fprintf(file_ptr, ".\n.\n.\n");
      fprintf(file_ptr, "%c = %3d:", indices[0], i);
      for (int j = 0; j < shape[1]; j++) {
        fprintf(file_ptr, "%15.3f", (float)a[i * stride[0] + j * stride[1]]);
        if (j == kMaxElement && j < shape[1]) {
          std::string efmt = " ... " + element_fmt;
          fprintf(file_ptr, efmt.c_str(), (float)a[i * stride[0] + (shape[1] - 1) * stride[1]]);
          break;
        }
      }
      break;
    }
  }
}


template <typename T>
__host__ __device__ void print_mat(const char* name, T* a, const char indices[2], const int shape[2], const int stride[2]) {
  constexpr int digits = 10;
  constexpr char hformat[] = "%10d";
  constexpr char eformat[] = "%10.2f";

  // Header
  printf("%s\n", name);
  int num_chars = 8 + (digits + 1) * shape[1];

  for (int i = 0; i < num_chars; i++) {
    printf("-");
  }
  printf("\n");

  printf("      %c: ", indices[1]);
  for (int j = 0; j < shape[1]; j++) {
    printf(hformat, j);
  }
  printf("\n");

  for (int i = 0; i < num_chars; i++) {
    printf("-");
  }
  printf("\n");

  // Body
  for (int i = 0; i < shape[0]; i++) {
    printf("%c = %3d:", indices[0], i);
    for (int j = 0; j < shape[1]; j++) {
      printf(eformat, (float)(a[i * stride[0] + j * stride[1]]));
    }
    printf("\n");
  }
}
