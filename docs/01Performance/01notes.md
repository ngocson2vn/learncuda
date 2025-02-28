# Understanding Performance
Performance of a function on a given processor is limited by one of the following three factors;
- memory bandwidth
- math bandwidth
- latency

## Memory bandwidth and math bandwidth
Consider a simplified model where a function reads its input from memory, performs math
operations, then writes its output to memory. Say $T_{mem}$ time is spent in accessing memory,
$T_{math}$ time is spent performing math operations.

If we further assume that memory and math portions of different threads can be overlapped (applying software pipeline, we can make memory and math operations overlapped), the total time for the function is $\max(T_{mem},\; T_{math})$. The longer of the two times demonstrates what limits performance: If math time is longer we say that a function is math limited, if memory time is longer then it is memory limited.

How much time is spent in memory or math operations depends on both **the algorithm and its implementation**, as well as **the processor’s bandwidths**.

**Memory time** is equal to the number of bytes accessed in memory divided by the processor’s memory bandwidth.

**Math time** is equal to the number of operations divided by the processor's math bandwidth.

**arithmetic intensity:** the ratio of algorithm implementation operations **# ops** and the number of bytes accessed **# bytes**, is known as the algorithm's arithmetic intensity. <br/>
**arithmetic intensity = # ops / # bytes**

**ops:byte ratio:** the ratio of a processor's math and memory bandwidths. <br/>
**ops:byte ratio = $BW_{math}$ / $BW_{mem}$**

An algorithm is **math limited on a given processor** if **the algorithm's arithmetic intensity** is higher than the processor's **ops:byte** ratio. <br/>
Conversely, an algorithm is **memory limited** if its arithmetic intensity is lower than the processor's ops:byte ratio.

For example, A10: https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/a10/pdf/a10-datasheet.pdf
FP16 Tensor Core: 125 TF <br/>
FP32: 31.2 TF <br/>
Global Memory BW: 600 GB/s <br/>
FP16 ops:byte ratio = 125 / 0.6 = 208 <br/>
FP32 ops:byte ratio = 31.2 / 0.6 = 52 <br/>

## Latency
The arithmetic intensity and ops:byte ratio analysis assumes that a workload is sufficiently large to saturate a given processor's math and memory pipelines. However, if the workload is **not large enough**, or **does not have sufficient parallelism**, the processor will be under-utilized and performance will be limited by **latency**.  
For example, consider the launch of a single thread that will access 16 bytes and perform 16000 math operations. While the arithmetic intensity is 1000 FLOPS/B and the execution should be math-limited on a V100 GPU, creating only a single thread grossly under-utilizes the GPU, leaving nearly all of its math pipelines and execution resources idle. Furthermore, the arithmetic intensity calculation assumes that inputs and outputs are accessed from memory exactly once. It is not unusual for algorithm implementations to read input elements multiple times, which would effectively reduce arithmetic intensity. Thus, the arithmetic intensity is a first-order approximation; profiler information should be used if more accurate analysis is needed.

## Summary
The most likely performance limiter is:
- Latency if there is not sufficient parallelism
- Math if there is sufficient parallelism and algorithm arithmetic intensity is higher than the GPU ops:byte ratio.
- Memory if there is sufficient parallelism and algorithm arithmetic intensity is lower than the GPU ops:byte ratio.

# Performance Profiling
A performance profiler must be able to answer the following questions:  
1. What is parallelism number?
2. What are min, avg, and max arithmetic intensities?
