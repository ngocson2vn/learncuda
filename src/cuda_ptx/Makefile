NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64
NVCC_FLAGS = -std=c++17 -g -O3 --compiler-options=-fno-inline --expt-relaxed-constexpr --disable-warnings
cuda_version := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda)
$(info NVCC: $(cuda_version))
cuda12_str := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda_12)
ifneq ("$(cuda12_str)", "")
	NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a
else
	$(error Feature 'elect' requires .target sm_90 or higher)
endif


.SUFFIXES: .o .cu

main: main.o
	g++ -L$(CUDA_LIB) -o main main.o -lcudart -ldl

main.o: main.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -c main.cu

clean:
	rm -rvf *.o main
