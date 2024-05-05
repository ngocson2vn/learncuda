#pragma once

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

int* mat_add(int* MatA, int* MatB, const int N);

#ifdef __cplusplus
}
#endif