# wgmma.mma_async
https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async

For example, `wgmma.mma_async.sync.aligned.m64n64k16.f16.f16.f16`  
The mandatory `.sync` qualifier indicates that wgmma.mma_async instruction causes the executing thread to wait until all threads in the warp execute the same wgmma.mma_async instruction before resuming execution.

The mandatory `.aligned` qualifier indicates that all threads in the warpgroup must execute the same wgmma.mma_async instruction. In conditionally executed code, a wgmma.mma_async instruction should only be used if it is known that all threads in the warpgroup evaluate the condition identically, otherwise behavior is undefined.
