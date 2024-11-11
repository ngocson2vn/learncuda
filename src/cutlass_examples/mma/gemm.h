#pragma once

#include <thrust/host_vector.h>

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
test_mma(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     thrust::host_vector<TA> const& h_A, int ldA,
     thrust::host_vector<TB> const& h_B, int ldB,
     Beta beta,
     thrust::host_vector<TC>      & h_C, int ldC);
