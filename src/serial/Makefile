NVCC = /usr/local/cuda/bin/nvcc
CUDA_INCLUDE = /usr/local/cuda/include
CUDA_LIB = /usr/local/cuda/lib64
NVCC_FLAGS = -std=c++17 -g -O3 --compiler-options=-fno-inline --expt-relaxed-constexpr --disable-warnings
NVCC_FLAGS += -gencode arch=compute_86,code=sm_86 -keep
cuda_version := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda)
$(info NVCC: $(cuda_version))
# cuda12_str := $(shell /usr/local/cuda/bin/nvcc --version | grep cuda_12)
# ifneq ("$(cuda12_str)", "")
# 	NVCC_FLAGS += -gencode arch=compute_90a,code=sm_90a
# else
# 	$(error Feature 'elect' requires .target sm_90 or higher)
# endif


.SUFFIXES: .o .cu

main: main.o
	g++ -L$(CUDA_LIB) -o main main.o -lcudart -ldl

main.o: main.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -c main.cu

test: test.o
	g++ -L$(CUDA_LIB) -o test test.o -lcudart -ldl

test.o: test.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -c test.cu
	# clang++ -std=c++17 -O0 -fno-inline --cuda-path=/usr/local/cuda --cuda-gpu-arch=sm_75 -S -emit-llvm test.cu


clean:
	ls | grep -v Makefile | grep -v helper | grep -E '*.cubin|*.cuda|*.cpp1|*.module|*.cpp4|*.fatbin' | xargs rm -vf
