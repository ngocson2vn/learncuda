# cuLaunchKernel and kernelParams
```C++
CUresult CUDAAPI cuLaunchKernel(CUfunction f,
                                unsigned int gridDimX,
                                unsigned int gridDimY,
                                unsigned int gridDimZ,
                                unsigned int blockDimX,
                                unsigned int blockDimY,
                                unsigned int blockDimZ,
                                unsigned int sharedMemBytes,
                                CUstream hStream,
                                void **kernelParams,
                                void **extra);

/**
 * \brief Launches a CUDA function where thread blocks can cooperate and synchronize as they execute
 *
 * Invokes the kernel \p f on a \p gridDimX x \p gridDimY x \p gridDimZ
 * grid of blocks. Each block contains \p blockDimX x \p blockDimY x
 * \p blockDimZ threads.
 *
 * \p sharedMemBytes sets the amount of dynamic shared memory that will be
 * available to each thread block.
 *
 * The device on which this kernel is invoked must have a non-zero value for
 * the device attribute ::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH.
 *
 * The total number of blocks launched cannot exceed the maximum number of blocks per
 * multiprocessor as returned by ::cuOccupancyMaxActiveBlocksPerMultiprocessor (or
 * ::cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags) times the number of multiprocessors
 * as specified by the device attribute ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT.
 *
 * The kernel cannot make use of CUDA dynamic parallelism.
 *
 * Kernel parameters must be specified via \p kernelParams.  If \p f
 * has N parameters, then \p kernelParams needs to be an array of N
 * pointers.  Each of \p kernelParams[0] through \p kernelParams[N-1]
 * must point to a region of memory from which the actual kernel
 * parameter will be copied.  The number of kernel parameters and their
 * offsets and sizes do not need to be specified as that information is
 * retrieved directly from the kernel's image.
 *
```
