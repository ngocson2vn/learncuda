NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64

CUTLASS_INCLUDE = ../../../cutlass/include
CUTLASS_UTIL = ../../../cutlass/tools/util/include

# NVCC_FLAGS = -std=c++17 -g -O0 --compiler-options=-fno-inline --expt-relaxed-constexpr -gencode arch=compute_75,code=sm_75 -gencode arch=compute_86,code=sm_86
NVCC_FLAGS = -std=c++17 -g -O0 --compiler-options=-fno-inline --expt-relaxed-constexpr
NVCC_FLAGS += --disable-warnings
cuda_version := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda)
$(info NVCC: $(cuda_version))
cuda12_str := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda_12)
ifneq ("$(cuda12_str)", "")
	NVCC_FLAGS +=  -gencode arch=compute_90a,code=sm_90a -D__CUDA_ARCH_FEAT_SM90_ALL
endif

test: cutlass test.o
	clang++ -ggdb -O0 -L$(CUDA_LIB) -lcudart -o test test.o && echo && ./test

main: cutlass sgemm_1.o sgemm_2.o main.o
	g++ -L$(CUDA_LIB) -lcudart -o main main.o sgemm_1.o sgemm_2.o

sgemm_1.o: sgemm_1.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -c sgemm_1.cu

sgemm_2.o: sgemm_2.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -c sgemm_2.cu

main.o: main.cc
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -c main.cc

CXX_FLAGS = -std=c++17 -ggdb -O0 -fno-inline -fno-limit-debug-info
test.o: test.cc
	clang++ $(CXX_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -c test.cc

xtest: test.cc
	/usr/bin/clang++ $(CXX_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -Xclang -ast-print -fsyntax-only test.cc > test.cxx

# wgmma on Hopper GPUs
hopper: wgmma_sm90.o
	g++ -L$(CUDA_LIB) -lcudart -o hopper wgmma_sm90.o

wgmma_sm90.o: wgmma_sm90.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL) -c wgmma_sm90.cu

cutlass:
	git submodule update --init --recursive

clean:
	rm -rvf *.o main test
