NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUTLASS_INCLUDE = ../../../cutlass/include
CUTLASS_UTIL = ../../../cutlass/tools/util/include
EXAMPLE_UTIL = ../

CXX_FLAGS = -std=c++17 -ggdb -O0 -fno-inline -fno-limit-debug-info


layout1: main1
	./main1 && ./pdf.sh

main1: cutlass main1.o
	clang++ -ggdb -O0 -L$(CUDA_LIB) -lcudart -o main1 main1.o

main1.o: main1.cc
	clang++ $(CXX_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -I$(EXAMPLE_UTIL) -c main1.cc

layout2: main2
	./main2 && ./pdf.sh

main2: cutlass main2.o
	clang++ -ggdb -O0 -L$(CUDA_LIB) -lcudart -o main2 main2.o

main2.o: main2.cc
	clang++ $(CXX_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -I$(EXAMPLE_UTIL) -c main2.cc

cutlass:
	git submodule update --init --recursive

clean:
	rm -rvf *.o main1 main2 result_layout*
