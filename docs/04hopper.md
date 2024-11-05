# Thread Block Clusters
https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#thread-block-clusters
https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/#thread_block_clusters

NVIDIA Hopper Architecture adds a new optional level of hierarchy, Thread Block Clusters, that allows for further possibilities when parallelizing applications. A thread block can read from, write to, and perform atomics in shared memory of other thread blocks within its cluster. This is known as Distributed Shared Memory. As demonstrated in the CUDA C++ Programming Guide, there are applications that cannot fit required data within shared memory and must use global memory instead. Distributed shared memory can act as an intermediate step between these two options.