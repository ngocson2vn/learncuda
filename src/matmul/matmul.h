#include <device_launch_parameters.h>

template <typename T, int BLOCK_SIZE>
T* matmul(T* MatA, T* MatB, const dim3& dimsA, const dim3& dimsB);

template <typename T, int BLOCK_SIZE>
T* smem_matmul(T* MatA, T* MatB, const dim3& dimsA, const dim3& dimsB);
