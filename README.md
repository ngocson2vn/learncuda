# learncuda
A repo for learning CUDA.
<br/>

**CUTLASS dependency**:
```Bash
git submodule update --init --recursive
```

# Debug cutlass
```Bash
mv cutlass_debug cutlass
cd src/cutlass_examples/hopper/
make
```

# Highlights
| Kernel     | Source code                                      |
|----------- |--------------------------------------------------|
| TMA        | [TMA](./src/cuda_tma/)                           |
| WGMMA      | [Hopper](./src/matmul_wgmma/hopper/)             |
| MMA v5     | [Blackwell](./src/matmul_wgmma/blackwell/)       |

