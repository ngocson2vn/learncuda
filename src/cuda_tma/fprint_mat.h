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
void fprint_mat(FILE* file_ptr, const char* name, T* a, const dim3& dims) {
  // Header
  fprintf(file_ptr, "%s\n", name);
  int num_chars = 8 + 7 * dims.y;
  fprintf(file_ptr, "%s\n", std::string(num_chars, '-').c_str());
  fprintf(file_ptr, "      j:");
  for (size_t j = 0; j < dims.y; j++) {
    fprintf(file_ptr, " %6d", j);
  }
  fprintf(file_ptr, "\n%s\n", std::string(num_chars, '-').c_str());

  // Body
  for (size_t i = 0; i < dims.x; i++) {
    fprintf(file_ptr, "i = %3d:", i);
    for (size_t j = 0; j < dims.y; j++) {
      fprintf(file_ptr, " %6.2f", (float)a[i + j * dims.x]);
      if (j == kMaxElement && j < dims.y) {
        fprintf(file_ptr, " ... %6.2f", (float)a[i + (dims.y - 1) * dims.x]);
        break;
      }
    }
    fprintf(file_ptr, "\n");
    if (i == kMaxElement && i < dims.x) {
      i = dims.x - 1;
      fprintf(file_ptr, ".\n.\n.\n");
      fprintf(file_ptr, "i = %3d:", i);
      for (size_t j = 0; j < dims.y; j++) {
        fprintf(file_ptr, " %6.2f", (float)a[i + j * dims.x]);
        if (j == kMaxElement && j < dims.y) {
          fprintf(file_ptr, " ... %6.2f", (float)a[i + (dims.y - 1) * dims.x]);
          break;
        }
      }
      break;
    }
  }
}

template<>
void fprint_mat<int>(FILE* file_ptr, const char* name, int* a, const dim3& dims) {
  // Header
  fprintf(file_ptr, "%s\n", name);
  int num_chars = 8 + 11 * dims.y;
  fprintf(file_ptr, "%s\n", std::string(num_chars, '-').c_str());
  fprintf(file_ptr, "      j:");
  for (size_t j = 0; j < dims.y; j++) {
    fprintf(file_ptr, " %10d", j);
  }
  fprintf(file_ptr, "\n%s\n", std::string(num_chars, '-').c_str());

  // Body
  for (size_t i = 0; i < dims.x; i++) {
    fprintf(file_ptr, "i = %3d:", i);
    for (size_t j = 0; j < dims.y; j++) {
      fprintf(file_ptr, " %10d", a[i + j * dims.x]);
      if (j == kMaxElement && j < dims.y) {
        fprintf(file_ptr, " ... %10d", a[i + (dims.y - 1) * dims.x]);
        break;
      }
    }
    fprintf(file_ptr, "\n");
    if (i == kMaxElement && i < dims.x) {
      i = dims.x - 1;
      fprintf(file_ptr, ".\n.\n.\n");
      fprintf(file_ptr, "i = %3d:", i);
      for (size_t j = 0; j < dims.y; j++) {
        fprintf(file_ptr, " %10d", a[i + j * dims.x]);
        if (j == kMaxElement && j < dims.y) {
          fprintf(file_ptr, " ... %10d", a[i + (dims.y - 1) * dims.x]);
          break;
        }
      }
      break;
    }
  }
}
