#include <cstdlib>
#include <cstdio>
#include <cassert>
#include <iostream>

#include "gemm.h"
#include "cutlass/util/print_error.hpp"

constexpr static int kMaxElement = 100;

template <typename T>
void print_mat(T* a, const char* name, const dim3& dims) {
  printf("%s\n", name);
  printf("==============================================================\n");
  for (size_t i = 0; i < dims.x; i++) {
    printf("i = %-3d:", i);
    for (size_t j = 0; j < dims.y; j++) {
      printf(" %-6.1f", a[i * dims.y + j]);
      if (j == kMaxElement && j < dims.y) {
        printf(" ... %-6.1f", a[i * dims.y + (dims.y - 1)]);
        break;
      }
    }
    printf("\n");
    if (i == kMaxElement && i < dims.x) {
      i = dims.x - 1;
      printf(".\n.\n.\n");
      printf("i = %-3d:", i);
      for (size_t j = 0; j < dims.y; j++) {
        printf(" %-6.1f", a[i * dims.y + j]);
        if (j == kMaxElement && j < dims.y) {
          printf(" ... %-6.1f", a[i * dims.y + (dims.y - 1)]);
          break;
        }
      }
      break;
    }
  }
}

int main(int argc, char** argv) {
  int m = 8;
  if (argc >= 2)
    sscanf(argv[1], "%d", &m);

  int n = 8;
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
  thrust::host_vector<TC> h_C(m*n);

  // for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>( 2*(rand() / double(RAND_MAX)) - 1 );
  // for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>( 2*(rand() / double(RAND_MAX)) - 1 );
  for (int j = 0; j < m*k; ++j) h_A[j] = static_cast<TA>(0.5);
  for (int j = 0; j < n*k; ++j) h_B[j] = static_cast<TB>(2);
  for (int j = 0; j < m*n; ++j) h_C[j] = static_cast<TC>(-1);



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

  // Run once
  test_mma(transA, transB, m, n, k,
       alpha,
       h_A, ldA,
       h_B, ldB,
       beta,
       h_C, ldC);

  printf("\n");
  print_mat(h_A.data(), "h_A", dim3(m, k, 1));
  printf("\n");

  printf("\n");
  print_mat(h_B.data(), "h_B", dim3(k, n, 1));
  printf("\n");

  print_mat(h_C.data(), "h_C", dim3(m, n, 1));
  // Timing iterations
  // double gflops = (2.0*m*n*k) * 1e-9;

  // const int timing_iterations = 100;
  // GPU_Clock timer;
  // timer.start();
  // for (int i = 0; i < timing_iterations; ++i) {
  //   gemm(transA, transB, m, n, k,
  //        alpha,
  //        d_A.data().get(), ldA,
  //        d_B.data().get(), ldB,
  //        beta,
  //        d_C.data().get(), ldC);
  // }
  // double cute_time = timer.seconds() / timing_iterations;
  // CUTE_CHECK_LAST();
  // printf("CUTE_GEMM:     [%6.1f]GFlop/s  (%6.4f)ms\n", gflops / cute_time, cute_time*1000);
  return 0;
}
