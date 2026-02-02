CUDA_INCLUDE = /usr/local/cuda-12.8/include
NVCC=/usr/local/cuda-12.8/bin/nvcc
CXX=/opt/tiger/cpp_tools/x86_64_x86_64_clang_2020/bin/clang++
NVCC_FLAGS=-std=c++17 -O3 -rdc=false -keep

# Suppress note: The ABI for passing parameters with 64-byte alignment has changed in GCC 4.6
NVCC_FLAGS+= -Xcompiler -Wno-psabi

NVCC_FLAGS+= --generate-code=arch=compute_100a,code=sm_100a
NVCC_FLAGS+= --expt-relaxed-constexpr -D__CUDA_ARCH_FEAT_SM100_ALL
NVCC_FLAGS+= -I$(CUDA_INCLUDE)

LDLIBS=-lcuda

.SUFFIXES: .o .cu

default: clean mmav5_with_tma

mmav5_with_tma: mmav5_with_tma.o
	$(NVCC) $(NVCC_FLAGS) $(LDFLAGS) -o $@ mmav5_with_tma.o $(LDLIBS)

mmav5_with_tma.o: device.h fprint_mat.h mmav5_with_tma.cu
	$(NVCC) -c $(NVCC_FLAGS) --keep -o "$@" mmav5_with_tma.cu

clean:
	rm -rvf *.o *.cubin *.fatbin* *.reg.c *.ii *.cudafe* *.module* *.ptx mmav5_with_tma test
