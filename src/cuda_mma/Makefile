NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64
NVCC_FLAGS = -std=c++17 -g -O3 --compiler-options=-fno-inline --expt-relaxed-constexpr --disable-warnings
NVCC_FLAGS += -gencode arch=compute_75,code=sm_75
cuda_version := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda)
$(info NVCC: $(cuda_version))
cuda12_str := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda_12)
ifneq ("$(cuda12_str)", "")
	NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a
endif


.SUFFIXES: .o .cu

hello_mma: hello_mma.o
	g++ -L$(CUDA_LIB) -o hello_mma hello_mma.o -lcudart -ldl

hello_mma.o: hello_mma.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -c hello_mma.cu

clean:
	rm -rvf *.o hello_mma
