CUDA_VERSION = 11.4
NVCC = /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda-$(CUDA_VERSION)/include
CXX = /usr/bin/clang++

NVCC_FLAGS = -std=c++17 -g -O3 -keep -v
NVCC_FLAGS += --compiler-options=-fno-inline --expt-relaxed-constexpr
NVCC_FLAGS += --disable-warnings
NVCC_FLAGS += -gencode arch=compute_75,code=sm_75
cuda_version := $(shell /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc --version | grep cuda)
$(info NVCC: $(cuda_version))
cuda12_str := $(shell /usr/local/cuda-$(CUDA_VERSION)/bin/nvcc --version | grep cuda_12)
ifneq ("$(cuda12_str)", "")
	NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a
endif

all:
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -c device.cu -o device.o
	$(CXX) -I$(CUDA_INCLUDE) -S -emit-llvm -o main.ll main.cc

clean:
	ls | grep -v Makefile | grep -v helper | grep -E '*.cubin|*.cuda|*.cpp1|*.module|*.cpp4|*.fatbin' | xargs rm -vf
	ls -lrt
