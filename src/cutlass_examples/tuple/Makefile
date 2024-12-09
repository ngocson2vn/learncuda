NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUTLASS_INCLUDE = ../../../cutlass/include
CUTLASS_UTIL = ../../../cutlass/tools/util/include

all: main
	./main

main: cutlass main.o
	clang++ -ggdb -O0 -L$(CUDA_LIB) -lcudart -o main main.o && echo && ./main

CXX_FLAGS = -std=c++17 -ggdb -O0 -fno-inline -fno-limit-debug-info
main.o: main.cc
	clang++ $(CXX_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -c main.cc

cutlass:
	git submodule update --init --recursive

clean:
	rm -rvf *.o main
