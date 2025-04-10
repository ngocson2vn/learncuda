n = 511

main = """
#include <cuda_runtime.h>
#include <stdio.h>

int const n = {n};

__global__ void testKernel(int n, {params}) {{
  // Dummy kernel
  printf("testKernel with params size %d bytes\\n", n);
}}

void launch_test_kernel(float* params[n]) {{
  testKernel<<<1, 1>>>(n, {args});
}}

int main() {{
  cudaError_t err;

  // Transfer data from host to device
  float* params[n];
  const int m = 10;
  for (int i = 0; i < n; i++) {{
    float* p = new float[m];
    for (int j = 0; j < m; j++) {{
      p[j] = 1.0 * j;
    }}

    float* d_p;
    cudaMalloc((void**)&d_p, m * sizeof(float));
    cudaMemcpyAsync(d_p, p, m * sizeof(float), cudaMemcpyHostToDevice);
    params[i] = d_p;
  }}

  launch_test_kernel(params);
  cudaDeviceSynchronize();
  err = cudaGetLastError();
  if (err != cudaSuccess) {{
      printf("Failed at %d bytes: %s\\n", n, cudaGetErrorString(err));
  }} else {{
      printf("%d bytes succeeded\\n", n);
  }}
  return 0;
}}
"""

output = open("main.cu", "w")
params = "float* p0"
args = "params[0]"
for i in range(1, n):
  params += f", float* p{i}"
  args += f", params[{i}]"

output.write(main.format(n=n, params=params, args=args))
output.flush()
output.close()
