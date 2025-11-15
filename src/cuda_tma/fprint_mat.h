#pragma once

#include <cstdio>
#include <cstdlib>
#include <device_launch_parameters.h>

FILE* get_file_ptr(const char* file_path) {
  FILE *fptr;
  fptr = fopen(file_path, "w");
  if(fptr == NULL) {
    printf("Failed to open %s\n", file_path);
    exit(1);
  }

  return fptr;
}

constexpr static int kMaxElement = 1000000;

template <typename T>
void fprint_mat(FILE* file_ptr, const char* name, T* a, const int shape[2], const int stride[2]) {
  // Header
  fprintf(file_ptr, "%s\n", name);
  int num_chars = 8 + 7 * shape[1];
  fprintf(file_ptr, "%s\n", std::string(num_chars, '-').c_str());
  fprintf(file_ptr, "      j:");
  for (size_t j = 0; j < shape[1]; j++) {
    fprintf(file_ptr, " %6d", j);
  }
  fprintf(file_ptr, "\n%s\n", std::string(num_chars, '-').c_str());

  // Body
  for (size_t i = 0; i < shape[0]; i++) {
    fprintf(file_ptr, "i = %3d:", i);
    for (size_t j = 0; j < shape[1]; j++) {
      fprintf(file_ptr, " %6.2f", (float)a[i * stride[0] + j * stride[1]]);
      if (j == kMaxElement && j < shape[1]) {
        fprintf(file_ptr, " ... %6.2f", (float)a[i * stride[0] + (shape[1] - 1) * stride[1]]);
        break;
      }
    }
    fprintf(file_ptr, "\n");
    if (i == kMaxElement && i < shape[0]) {
      i = shape[0] - 1;
      fprintf(file_ptr, ".\n.\n.\n");
      fprintf(file_ptr, "i = %3d:", i);
      for (size_t j = 0; j < shape[1]; j++) {
        fprintf(file_ptr, " %6.2f", (float)a[i * stride[0] + j * stride[1]]);
        if (j == kMaxElement && j < shape[1]) {
          fprintf(file_ptr, " ... %6.2f", (float)a[i * stride[0] + (shape[1] - 1) * stride[1]]);
          break;
        }
      }
      break;
    }
  }
}

template<>
void fprint_mat<int>(FILE* file_ptr, const char* name, int* a, const int shape[2], const int stride[2]) {
  constexpr int digits = 3;
  std::string format = " %" + std::to_string(digits) + "d";

  // Header
  fprintf(file_ptr, "%s\n", name);
  int num_chars = 8 + (digits + 1) * shape[1];
  fprintf(file_ptr, "%s\n", std::string(num_chars, '-').c_str());
  fprintf(file_ptr, "      j:");
  for (size_t j = 0; j < shape[1]; j++) {
    fprintf(file_ptr, format.c_str(), j);
  }
  fprintf(file_ptr, "\n%s\n", std::string(num_chars, '-').c_str());

  // Body
  for (size_t i = 0; i < shape[0]; i++) {
    fprintf(file_ptr, "i = %3d:", i);
    for (size_t j = 0; j < shape[1]; j++) {
      fprintf(file_ptr, format.c_str(), a[i * stride[0] + j * stride[1]]);
      if (j == kMaxElement && j < shape[1]) {
        format = " ..." + format;
        fprintf(file_ptr, format.c_str(), a[i * stride[0] + (shape[1] - 1) * stride[1]]);
        break;
      }
    }
    fprintf(file_ptr, "\n");
    if (i == kMaxElement && i < shape[0]) {
      i = shape[0] - 1;
      fprintf(file_ptr, ".\n.\n.\n");
      fprintf(file_ptr, "i = %3d:", i);
      for (size_t j = 0; j < shape[1]; j++) {
        fprintf(file_ptr, format.c_str(), a[i * stride[0] + j * stride[1]]);
        if (j == kMaxElement && j < shape[1]) {
          format = " ..." + format;
          fprintf(file_ptr, format.c_str(), a[i * stride[0] + (shape[1] - 1) * stride[1]]);
          break;
        }
      }
      break;
    }
  }
}
