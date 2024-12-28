# SASS (Streaming Assembler)
In the context of NVIDIA CUDA programming, SASS (Streaming Assembler) refers to the low-level assembly language or Instruction Set Architecture (ISA) used by NVIDIA GPUs. SASS instructions are the hardware-specific binary instructions executed directly by the GPU's processing cores.
<br/><br/>

# Compile PTX to SASS
To compile a **PTX** file to **SASS**, you use the NVIDIA CUDA **`ptxas`** tool, which is a component of the NVIDIA CUDA toolkit. The `ptxas` tool takes PTX (Parallel Thread Execution) code as input and generates SASS (Streaming Assembler) machine code specific to a target GPU architecture.

### Steps to Compile a PTX File to SASS

#### 1. **Generate the PTX File**
First, compile your CUDA source code (`.cu`) into a PTX file using the **`nvcc`** compiler:
```bash
nvcc -ptx -arch=sm_80 source.cu -o output.ptx
```
- `-ptx`: Specifies that the output should be PTX.
- `-arch=sm_XX`: Sets the target compute architecture (e.g., `sm_80` for NVIDIA Ampere GPUs).

This will generate a `output.ptx` file.

#### 2. **Compile PTX to SASS Using `ptxas`**
Next, use the `ptxas` tool to compile the PTX file into SASS:
```bash
ptxas -arch=sm_80 output.ptx -o output.cubin
```
- `-arch=sm_XX`: Specifies the target GPU architecture.
- `-o output.cubin`: Produces a compiled binary object (`.cubin`) containing the SASS instructions.

The `output.cubin` file contains the SASS code, which can be executed directly on the GPU.

#### 3. **Disassemble SASS for Debugging (Optional)**
To view the generated SASS code in human-readable form, use the **`nvdisasm`** tool:
```bash
nvdisasm output.cubin > sass_output.txt
```
This will generate a disassembled version of the SASS code in `sass_output.txt`.

### Example Workflow
1. Generate PTX:
   ```bash
   nvcc -ptx -arch=sm_80 my_kernel.cu -o my_kernel.ptx
   ```
2. Compile PTX to SASS:
   ```bash
   ptxas -arch=sm_80 my_kernel.ptx -o my_kernel.cubin
   ```
3. Inspect SASS:
   ```bash
   nvdisasm my_kernel.cubin > my_kernel_sass.txt
   ```

### Notes
- Ensure that the **CUDA Toolkit version** supports the target GPU architecture (e.g., `sm_80` for Ampere GPUs).
- Use the **correct `sm_XX` architecture flag** for your target GPU.
- If you want to include debugging information, use the `--keep` and `-G` flags during the initial `nvcc` compilation.

### When Do You Need This?
- **Low-Level Debugging**: When optimizing for performance and inspecting how PTX translates to hardware-specific SASS instructions.
- **Performance Tuning**: Understanding memory usage, register allocation, and warp execution at the hardware level.

By using these tools and steps, you can efficiently generate and inspect SASS for advanced CUDA programming needs.
<br/><br/>

# How to execute a `.cubin` file
```C++
#include <cuda.h>
#include <iostream>

int main() {
    // Initialize CUDA Driver API
    CUdevice device;
    CUcontext context;
    CUmodule module;
    CUfunction kernel;

    // Initialize the CUDA driver
    cuInit(0);

    // Get the first CUDA device
    cuDeviceGet(&device, 0);

    // Create a CUDA context
    cuCtxCreate(&context, 0, device);

    // Load the .cubin file
    cuModuleLoad(&module, "output.cubin");

    // Retrieve the kernel function from the module
    cuModuleGetFunction(&kernel, module, "kernel_function_name");

    // Set up kernel launch parameters
    int gridDimX = 1, gridDimY = 1, gridDimZ = 1;
    int blockDimX = 32, blockDimY = 1, blockDimZ = 1;

    // Launch the kernel (assuming no kernel arguments)
    void* args[] = { /* Kernel arguments go here */ };
    cuLaunchKernel(kernel,
                   gridDimX, gridDimY, gridDimZ,  // Grid dimensions
                   blockDimX, blockDimY, blockDimZ,  // Block dimensions
                   0, nullptr,  // Shared memory size and stream
                   args,  // Kernel arguments
                   nullptr);

    // Clean up
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
```
