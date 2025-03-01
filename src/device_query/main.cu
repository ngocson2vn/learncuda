/* Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/* This sample queries the properties of the CUDA devices present in the system
 * via CUDA Runtime API. */

// std::system includes

#include <cuda_runtime.h>
#include "helper_cuda.h"

#include <iostream>
#include <memory>
#include <string>

#define LOG(level, format, ...)              \
  if (level > 0) {                           \
    fprintf(stdout, format, ##__VA_ARGS__);    \
  }

int *pArgc = NULL;
char **pArgv = NULL;

struct DeviceProperties {
  std::string name;
  int computeMajor;
  int computeMinor;
  double peakFp16TensorCoreFlops;
  double peakMemoryBwBytesPerSec;
};

template <typename T>
std::string format(std::string key, T value) {
  return "\"" + key + "\": " + std::to_string(value);
}

std::string format(std::string key, std::string value) {
  return "\"" + key + "\": " + "\"" + value + "\"";
}

std::string createJson(const DeviceProperties& devProp) {
  std::string retJson = "{";
  retJson += format("name", devProp.name);
  retJson += ", " + format("computeMajor", devProp.computeMajor);
  retJson += ", " + format("computeMinor", devProp.computeMinor);
  retJson += ", " + format("peakFp16TensorCoreFlops", devProp.peakFp16TensorCoreFlops);
  retJson += ", " + format("peakMemoryBwBytesPerSec", devProp.peakMemoryBwBytesPerSec);
  retJson += "}";

  return retJson;
}

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
  pArgc = &argc;
  pArgv = argv;

  int level = 0;
  if (argc > 1) {
    std::string mode(pArgv[1]);
    if (mode == "verbose") {
      level = 1;
    }
  }

  DeviceProperties dp;

  LOG(level, "%s Starting...\n\n", argv[0]);
  LOG(level, " CUDA Device Query (Runtime API) version (CUDART static linking)\n\n");

  int deviceCount = 0;
  cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

  if (error_id != cudaSuccess) {
    LOG(level, "cudaGetDeviceCount returned %d\n-> %s\n",
           static_cast<int>(error_id), cudaGetErrorString(error_id));
    LOG(level, "Result = FAIL\n");
    exit(EXIT_FAILURE);
  }

  // This function call returns 0 if there are no CUDA capable devices.
  if (deviceCount == 0) {
    LOG(level, "There are no available device(s) that support CUDA\n");
  } else {
    LOG(level, "Detected %d CUDA Capable device(s)\n", deviceCount);
  }

  int dev, driverVersion = 0, runtimeVersion = 0;
  cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, dev);

  LOG(level, "\nDevice %d: \"%s\"\n", dev, deviceProp.name);
  dp.name = deviceProp.name;

  // Console log
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  LOG(level, "  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n",
          driverVersion / 1000, (driverVersion % 100) / 10,
          runtimeVersion / 1000, (runtimeVersion % 100) / 10);

  LOG(level, "  CUDA Capability Major/Minor version number:    %d.%d\n",
          deviceProp.major, deviceProp.minor);
  dp.computeMajor = deviceProp.major;
  dp.computeMinor = deviceProp.minor;

  char msg[256];
  snprintf(msg, sizeof(msg),
            "  Total amount of global memory:                 %.0f MBytes "
            "(%llu bytes)\n",
            static_cast<float>(deviceProp.totalGlobalMem / 1048576.0f),
            (unsigned long long)deviceProp.totalGlobalMem);
  LOG(level, "%s", msg);

  LOG(level, "  (%03d) Multiprocessors, (%03d) CUDA Cores/MP:    %d CUDA Cores\n",
          deviceProp.multiProcessorCount,
          _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
          _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) *
              deviceProp.multiProcessorCount);
  LOG(level, 
      "  GPU Max Clock rate:                            %.0f MHz (%0.2f GHz)\n",
      deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);

  // This is supported in CUDA 5.0 (runtime API device properties)
  LOG(level, "  Memory Clock rate:                             %.0f Mhz\n",
          deviceProp.memoryClockRate * 1e-3f);
  LOG(level, "  Memory Bus Width:                              %d-bit\n",
          deviceProp.memoryBusWidth);

  if (deviceProp.l2CacheSize) {
    LOG(level, "  L2 Cache Size:                                 %d bytes\n",
            deviceProp.l2CacheSize);
  }

  LOG(level, 
      "  Maximum Texture Dimension Size (x,y,z)         1D=(%d), 2D=(%d, "
      "%d), 3D=(%d, %d, %d)\n",
      deviceProp.maxTexture1D, deviceProp.maxTexture2D[0],
      deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0],
      deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
  LOG(level, 
      "  Maximum Layered 1D Texture Size, (num) layers  1D=(%d), %d layers\n",
      deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1]);
  LOG(level, 
      "  Maximum Layered 2D Texture Size, (num) layers  2D=(%d, %d), %d "
      "layers\n",
      deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1],
      deviceProp.maxTexture2DLayered[2]);

  LOG(level, "  Total amount of constant memory:               %zu bytes\n",
          deviceProp.totalConstMem);
  LOG(level, "  Total amount of shared memory per block:       %zu bytes\n",
          deviceProp.sharedMemPerBlock);
  LOG(level, "  Total amount of dynamic shared memory:         %zu bytes\n",
          deviceProp.sharedMemPerBlockOptin);
  LOG(level, "  Total shared memory per multiprocessor:        %zu bytes\n",
          deviceProp.sharedMemPerMultiprocessor);
  LOG(level, "  Total number of registers available per block: %d\n",
          deviceProp.regsPerBlock);
  LOG(level, "  Warp size:                                     %d\n",
          deviceProp.warpSize);
  LOG(level, "  Maximum number of threads per multiprocessor:  %d\n",
          deviceProp.maxThreadsPerMultiProcessor);
  LOG(level, "  Maximum number of threads per block:           %d\n",
          deviceProp.maxThreadsPerBlock);
  LOG(level, "  Max dimension size of a thread block (x,y,z): (%d, %d, %d)\n",
          deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1],
          deviceProp.maxThreadsDim[2]);
  LOG(level, "  Max dimension size of a grid size    (x,y,z): (%d, %d, %d)\n",
          deviceProp.maxGridSize[0], deviceProp.maxGridSize[1],
          deviceProp.maxGridSize[2]);
  LOG(level, "  Maximum memory pitch:                          %zu bytes\n",
          deviceProp.memPitch);
  LOG(level, "  Texture alignment:                             %zu bytes\n",
          deviceProp.textureAlignment);
  LOG(level, 
      "  Concurrent copy and kernel execution:          %s with %d copy "
      "engine(s)\n",
      (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
  LOG(level, "  Run time limit on kernels:                     %s\n",
          deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
  LOG(level, "  Integrated GPU sharing Host Memory:            %s\n",
          deviceProp.integrated ? "Yes" : "No");
  LOG(level, "  Support host page-locked memory mapping:       %s\n",
          deviceProp.canMapHostMemory ? "Yes" : "No");
  LOG(level, "  Alignment requirement for Surfaces:            %s\n",
          deviceProp.surfaceAlignment ? "Yes" : "No");
  LOG(level, "  Device has ECC support:                        %s\n",
          deviceProp.ECCEnabled ? "Enabled" : "Disabled");
  LOG(level, "  Device supports Unified Addressing (UVA):      %s\n",
          deviceProp.unifiedAddressing ? "Yes" : "No");
  LOG(level, "  Device supports Managed Memory:                %s\n",
          deviceProp.managedMemory ? "Yes" : "No");
  LOG(level, "  Device supports Compute Preemption:            %s\n",
          deviceProp.computePreemptionSupported ? "Yes" : "No");
  LOG(level, "  Supports Cooperative Kernel Launch:            %s\n",
          deviceProp.cooperativeLaunch ? "Yes" : "No");
  LOG(level, "  Supports MultiDevice Co-op Kernel Launch:      %s\n",
          deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No");
  LOG(level, "  Device PCI Domain ID / Bus ID / location ID:   %d / %d / %d\n",
          deviceProp.pciDomainID, deviceProp.pciBusID, deviceProp.pciDeviceID);

  // Estimate Tensor Cores per SM based on architecture
  int tensorCoresPerSM = 0;
  if (deviceProp.major == 7 && deviceProp.minor >= 0) { // Volta (7.0), Turing (7.5)
    tensorCoresPerSM = 8; // Volta/Turing
  } else if (deviceProp.major == 8) { // Ampere (8.x)
    tensorCoresPerSM = 4; // Ampere
  } else if (deviceProp.major == 9) {
    tensorCoresPerSM = 4; // Hopper
  } else {
    LOG(level, "  No Tensor Cores (pre-Volta architecture)\n");
  }

  int totalTensorCores = deviceProp.multiProcessorCount * tensorCoresPerSM;
  LOG(level, "  Tensor Cores per SM:                           %d\n", tensorCoresPerSM);
  LOG(level, "  Total Tensor Cores:                            %d\n", totalTensorCores);

  //===========================================================================================
  // Tensor Core
  //===========================================================================================
  // Manually define FLOPs per Tensor Core per Cycle (architecture-specific)
  float flopsPerTensorCorePerCycle = 0;
  if (deviceProp.major == 7 && deviceProp.minor == 0) { // Volta
    flopsPerTensorCorePerCycle = 64; // FP16, dense
  } else if (deviceProp.major == 7 && deviceProp.minor == 5) { // Turing
    flopsPerTensorCorePerCycle = 64; // FP16, dense
  } else if (deviceProp.major == 8) { // Ampere
    flopsPerTensorCorePerCycle = 256; // FP16, dense (512 with sparsity)
  } else if (deviceProp.major == 9) { // Hopper
    flopsPerTensorCorePerCycle = 956.5; // FP16, dense (1913 with sparsity)
  }

  LOG(level, "  FLOPs per Tensor Core per Cycle (FP16, dense): %d\n", int(flopsPerTensorCorePerCycle));

  // Calculate peak FP16 Tensor Core FLOPs
  double peakFlops = (double)deviceProp.multiProcessorCount * tensorCoresPerSM;
          peakFlops *= flopsPerTensorCorePerCycle * (deviceProp.clockRate * 1e3); // Hz
  LOG(level, "  Peak FP16 Tensor Core FLOPs (dense):           %d TFLOPs\n", int(peakFlops * 1e-12 + 0.5));
  dp.peakFp16TensorCoreFlops = peakFlops;
  //===========================================================================================

  // Peak Memory Bandwidth (GB/s) = (Memory Clock Rate (Hz) * Memory Bus Width (bits) * Number of Transfers per Clock) / 8
  double peakMemoryBw = (deviceProp.memoryClockRate * 1e3f * deviceProp.memoryBusWidth * 2) / (8 * 1e9f);
  LOG(level, "  Peak Memory Bandwidth (GB/s):                  %d GB/s\n", int(peakMemoryBw + 0.5));
  dp.peakMemoryBwBytesPerSec = peakMemoryBw * 1e9f;

  // ops:byte ratio
  double opsByteRatio = peakFlops / (peakMemoryBw * 1e9f);
  LOG(level, "  Peak FP16 Tensor Core ops:byte ratio:          %d\n", int(opsByteRatio + 0.5));

  const char *sComputeMode[] = {
    "Default (multiple host threads can use ::cudaSetDevice() with device "
    "simultaneously)",
    "Exclusive (only one host thread in one process is able to use "
    "::cudaSetDevice() with this device)",
    "Prohibited (no host thread can use ::cudaSetDevice() with this "
    "device)",
    "Exclusive Process (many threads in one process is able to use "
    "::cudaSetDevice() with this device)",
    "Unknown", NULL};
  LOG(level, "  Compute Mode:\n");
  LOG(level, "     < %s >\n", sComputeMode[deviceProp.computeMode]);


  // If there are 2 or more GPUs, query to determine whether RDMA is supported
  if (deviceCount >= 2) {
    cudaDeviceProp prop[64];
    int gpuid[64];  // we want to find the first two GPUs that can support P2P
    int gpu_p2p_count = 0;

    for (int i = 0; i < deviceCount; i++) {
      checkCudaErrors(cudaGetDeviceProperties(&prop[i], i));

      // Only boards based on Fermi or later can support P2P
      if (prop[i].major >= 2) {
        // This is an array of P2P capable GPUs
        gpuid[gpu_p2p_count++] = i;
      }
    }

    // Show all the combinations of support P2P GPUs
    int can_access_peer;

    if (gpu_p2p_count >= 2) {
      for (int i = 0; i < gpu_p2p_count; i++) {
        for (int j = 0; j < gpu_p2p_count; j++) {
          if (gpuid[i] == gpuid[j]) {
            continue;
          }
          checkCudaErrors(
              cudaDeviceCanAccessPeer(&can_access_peer, gpuid[i], gpuid[j]));
          LOG(level, "> Peer access from %s (GPU%d) -> %s (GPU%d) : %s\n",
                 prop[gpuid[i]].name, gpuid[i], prop[gpuid[j]].name, gpuid[j],
                 can_access_peer ? "Yes" : "No");
        }
      }
    }
  }

  // csv masterlog info
  // *****************************
  // exe and CUDA driver name
  LOG(level, "\n");
  std::string sProfileString = "deviceQuery, CUDA Driver = CUDART";
  char cTemp[16];

  // driver version
  sProfileString += ", CUDA Driver Version = ";
  snprintf(cTemp, sizeof(cTemp), "%d.%d", driverVersion / 1000,
           (driverVersion % 100) / 10);
  sProfileString += cTemp;

  // Runtime version
  sProfileString += ", CUDA Runtime Version = ";
  snprintf(cTemp, sizeof(cTemp), "%d.%d", runtimeVersion / 1000,
           (runtimeVersion % 100) / 10);
  sProfileString += cTemp;

  // Device count
  sProfileString += ", NumDevs = ";
  snprintf(cTemp, sizeof(cTemp), "%d", deviceCount);
  sProfileString += cTemp;
  sProfileString += "\n";
  LOG(level, "%s", sProfileString.c_str());

  LOG(level, "Result = PASS\n");

  if (level < 1) {
    std::cout << createJson(dp);
  }

  // finish
  exit(EXIT_SUCCESS);
}
