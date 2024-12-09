NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

CUTLASS_INCLUDE = ../../../cutlass/include
CUTLASS_UTIL = ../../../cutlass/tools/util/include

NVCC_FLAGS = -std=c++17 -g -O0 --compiler-options=-fno-inline --expt-relaxed-constexpr
NVCC_FLAGS += --disable-warnings
cuda_version := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda)
$(info NVCC: $(cuda_version))
cuda12_str := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda_12)
ifneq ("$(cuda12_str)", "")
	NVCC_FLAGS +=  -gencode arch=compute_90a,code=sm_90a -D__CUDA_ARCH_FEAT_SM90_ALL
endif

run: main
	./main

main: main.o
	g++ -L$(CUDA_LIB) -lcudart -o main main.o && echo

main.o: cutlass main.cc
	$(NVCC) $(NVCC_FLAGS) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -c main.cc

cutlass:
	git submodule update --init --recursive

clean:
	rm -rvf *.o main
