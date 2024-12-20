NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include


NVCC_FLAGS = -std=c++17 -g -O0
# NVCC_FLAGS += -keep
NVCC_FLAGS += --compiler-options=-fno-inline --expt-relaxed-constexpr
NVCC_FLAGS += --disable-warnings
NVCC_FLAGS += -gencode arch=compute_75,code=sm_75
cuda_version := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda)
$(info NVCC: $(cuda_version))
cuda12_str := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda_12)
ifneq ("$(cuda12_str)", "")
	NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a
endif

main: main.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -o main main.cu

clean:
	ls | grep -v main.cu | grep -v Makefile | xargs rm -vf
