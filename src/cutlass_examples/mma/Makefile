NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

CUTLASS_INCLUDE = ../../../cutlass/include
CUTLASS_UTIL = ../../../cutlass/tools/util/include

NVCC_FLAGS = -std=c++17 --expt-relaxed-constexpr -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90

all: main
	./main

main: sgemm_1.o main.o
	g++ -L$(CUDA_LIB) -lcudart -o main main.o sgemm_1.o

sgemm_1.o: sgemm_1.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -c sgemm_1.cu

main.o: main.cc
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -c main.cc

clean:
	rm -rvf *.o main
