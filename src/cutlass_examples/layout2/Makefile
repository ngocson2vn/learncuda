NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUTLASS_INCLUDE = ../../../cutlass/include

all: main
	./main

main: main.cc
	g++ -std=c++17 -g -O0 -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -o main main.cc

clean:
	rm -rvf *.o main
