#include <cstdio>
#include <random>
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
    fprintf(file_ptr, "i = %-3d:", i);
    for (size_t j = 0; j < dims.y; j++) {
      fprintf(file_ptr, " %6.1f", a[i + j * dims.x]);
      if (j == kMaxElement && j < dims.y) {
        fprintf(file_ptr, " ... %6.1f", a[i + (dims.y - 1) * dims.x]);
        break;
      }
    }
    fprintf(file_ptr, "\n");
    if (i == kMaxElement && i < dims.x) {
      i = dims.x - 1;
      fprintf(file_ptr, ".\n.\n.\n");
      fprintf(file_ptr, "i = %-3d:", i);
      for (size_t j = 0; j < dims.y; j++) {
        fprintf(file_ptr, " %6.1f", a[i + j * dims.x]);
        if (j == kMaxElement && j < dims.y) {
          fprintf(file_ptr, " ... %6.1f", a[i + (dims.y - 1) * dims.x]);
          break;
        }
      }
      break;
    }
  }
}

int main(int argc, char** argv) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_int_distribution<std::mt19937::result_type> dist3(0.0, 10.0); // distribution in range

  const int kBlockSize = 8;
  const int m = 3 * kBlockSize;
  const int n = 2 * kBlockSize;
  const dim3 dimsA(m, n); // rows(y): 2 x columns(x): 3
  float* MatA = new float[dimsA.x * dimsA.y];
  for (size_t i = 0; i < dimsA.y; i++) {
    for (size_t j = 0; j < dimsA.x; j++) {
      MatA[i * dimsA.x + j] = dist3(rng);
    }
  }

  FILE* file_ptr = get_file_ptr("output.txt");
  fprint_mat(file_ptr, "MatA", MatA, dim3(m, n, 1));
  fprintf(file_ptr, "\n\n");
  printf("Output file: output.txt\n");
}