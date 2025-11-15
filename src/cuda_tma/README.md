# NOTE
The latest TMA kernel is [main3.cu](./main3.cu).
```
TMA kernel launch: 4 blocks of 32 threads.
Each block operates on a quarter of matrix A along column dimension:
- Copy a quarter of matrix A to shared memory
- Update each element of the quarter residing in shared memory
- Transfer the updated quarter from shared memory back to global memory
|----------------------------|
|     quarter1: 16x32        |
|----------------------------|
|     quarter2: 16x32        |
|----------------------------|
|     quarter3: 16x32        |
|----------------------------|
|     quarter4: 16x32        |
|----------------------------|
```

## Prerequisites
```Bash
GPU: NVIDIA H100
Driver Version: >=535.129.03
CUDA Version: 12.4
```

## Build
```Bash
make
```

## Run
```Bash
./main3
```
Matrix A is dumped to `output.txt`.

