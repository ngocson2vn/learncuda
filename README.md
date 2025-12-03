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
| Kernel     | Source code                             |
|----------- |-----------------------------------------|
| TMA        | [cuda_tma](./src/cuda_tma/)             |
| WGMMA      | [matmul_wgmma](./src/matmul_wgmma/)     |

