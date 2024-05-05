#include <cstdio>
#include <cstdlib>

#include "cuda.h"

int main() {
  int deviceID = 0;
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, deviceID);
  printf("%-40s %s\n", "name:", prop.name);
  printf("%-40s %d\n", "multiProcessorCount:", prop.multiProcessorCount);
  printf("%-40s %ld\n", "totalGlobalMem:", prop.totalGlobalMem);
  printf("%-40s %d\n", "warpSize:", prop.warpSize);
  printf("%-40s %d\n", "maxThreadsPerBlock:", prop.maxThreadsPerBlock);
  printf("%-40s %d\n", "maxThreadsPerMultiProcessor:", prop.maxThreadsPerMultiProcessor);
  printf("%-40s %d\n", "asyncEngineCount:", prop.asyncEngineCount);
  printf("%-40s %d\n", "concurrentKernels:", prop.concurrentKernels);
  printf("%-40s %d\n", "integrated:", prop.integrated);
}