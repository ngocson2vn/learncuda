NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUTLASS_INCLUDE = ../../../cutlass/include
CUTLASS_UTIL = ../../../cutlass/tools/util/include
EXAMPLE_UTIL = ../

CXX_FLAGS = -std=c++17 -ggdb -O0 -fno-inline -fno-limit-debug-info

_ := $(shell git submodule update --init --recursive)

all: main
	./main

latex: layout
	pdflatex layout.tex && code layout.pdf

layout: main
	./main

main: main.o
	clang++ -ggdb -O0 -L$(CUDA_LIB) -lcudart -o main main.o && echo

main.o: main.cc
	clang++ $(CXX_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -I$(EXAMPLE_UTIL) -c main.cc


clean:
	rm -rvf *.o main
