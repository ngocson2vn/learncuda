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

int main(int argc, char** argv) {
  int m = 16;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 16;
  if (argc >= 3)
    sscanf(argv[2], "%d", &n);

  int k = 4;
  if (argc >= 4)
    sscanf(argv[3], "%d", &k);

  char transA = 'N';
  if (argc >= 5)
    sscanf(argv[4], "%c", &transA);

  char transB = 'T';
  if (argc >= 6)
    sscanf(argv[5], "%c", &transB);

  using TA = float;
  using TB = float;
  using TC = float;
  using TI = float;

  TI alpha = 1.0;
  TI beta  = 0.0;

  std::cout << "M = " << m << std::endl;
  std::cout << "N = " << n << std::endl;
  std::cout << "K = " << k << std::endl;
  std::cout << "C = A^" << transA << " B^" << transB << std::endl;

  thrust::host_vector<TA> h_A(m*k);
  thrust::host_vector<TB> h_B(n*k);
  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>(0.5);
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>(2);

  int ldA = 0, ldB = 0, ldC = m;

  if (transA == 'N') {
    ldA = m;
  } else if (transA == 'T') {
    ldA = k;
  } else {
    assert(false);
  }

  if (transB == 'N') {
    ldB = k;
  } else if (transB == 'T') {
    ldB = n;
  } else {
    assert(false);
  }

  printf("\n");
  print_mat("h_A", h_A.data(), dim3(m, k, 1));
  printf("\n");

  printf("\n");
  print_mat("h_B", h_B.data(), dim3(n, k, 1));
  printf("\n");

  float* h_C_cpu = matmul_nt_cpu(h_A.data(), h_A.size(), ldA, h_B.data(), h_B.size(), ldB);
  printf("\n");
  print_mat("h_C_cpu", h_C_cpu, dim3(m, n, 1));
  printf("\n");

  // Init cuda
  cute::device_init(0);

  //==============================================================
  // v1
  //==============================================================
  thrust::host_vector<TC> h_C_v1(m*n);
  for (int j = 0; j < m*n; ++j) h_C_v1[j] = static_cast<TC>(0.0);
  printf("\n");
  print_mat("h_C_v1 BEFORE", h_C_v1.data(), dim3(m, n, 1));
  printf("\n");

  v1::test_mma_v1(transA, transB, m, n, k,
       alpha,
       h_A, ldA,
       h_B, ldB,
       beta,
       h_C_v1, ldC);

  printf("\n");
  print_mat("h_C_v1 AFTER", h_C_v1.data(), dim3(m, n, 1));

  //==============================================================
  // v2
  //==============================================================
  // thrust::host_vector<TC> h_C_v2(m*n);
  // for (int j = 0; j < m*n; ++j) h_C_v2[j] = static_cast<TC>(0.0);
  // printf("\n");
  // print_mat("h_C_v2 BEFORE", h_C_v2.data(), dim3(m, n, 1));
  // printf("\n");

  // printf("\n");
  // v2::test_mma_v2(transA, transB, m, n, k,
  //      alpha,
  //      h_A, ldA,
  //      h_B, ldB,
  //      beta,
  //      h_C_v2, ldC);

  // printf("\n");
  // print_mat("h_C_v2 AFTER", h_C_v2.data(), dim3(m, n, 1));
  // printf("\n");

  return 0;
}
