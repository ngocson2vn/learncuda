#pragma once

#include <thrust/host_vector.h>

namespace v1 {
template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
test_mma_v1(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     thrust::host_vector<TA> const& h_A, int ldA,
     thrust::host_vector<TB> const& h_B, int ldB,
     Beta beta,
     thrust::host_vector<TC>      & h_C, int ldC);

} // namespace v1

namespace v2 {

template <class TA, class TB, class TC,
          class Alpha, class Beta>
void
test_mma_v2(char transA, char transB, int m, int n, int k,
     Alpha alpha,
     thrust::host_vector<TA> const& h_A, int ldA,
     thrust::host_vector<TB> const& h_B, int ldB,
     Beta beta,
     thrust::host_vector<TC>      & h_C, int ldC);

} // namespace v2
