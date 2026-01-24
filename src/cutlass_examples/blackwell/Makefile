CUDA_INCLUDE = /usr/local/cuda-12.8/include
CUTLASS_INCLUDE = ../../../cutlass/include
CUTLASS_UTIL_INCLUDE = ../../../cutlass/tools/util/include
CXX=/usr/local/cuda-12.8/bin/nvcc
CXXFLAGS=-std=c++17 -g -O0

# Suppress note: The ABI for passing parameters with 64-byte alignment has changed in GCC 4.6
CXXFLAGS+= -Xcompiler "-Wno-psabi -fno-inline-functions"
CXXFLAGS+= -lineinfo -keep
CXXFLAGS+= --generate-code=arch=compute_100a,code=sm_100a
CXXFLAGS+= --expt-relaxed-constexpr -D__CUDA_ARCH_FEAT_SM90_ALL
CXXFLAGS+= -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL_INCLUDE)
ALL_INCLUDE = -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL_INCLUDE)

LIB_DIRS=-L/usr/local/cuda-12.8/lib64
LDLIBS=-lcudart -lcuda

.SUFFIXES: .o .cu

default: clean mma_tma_sm100

mma_tma_sm100: clean mma_tma_sm100.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIB_DIRS) -o $@ mma_tma_sm100.o $(LDLIBS)

mma_tma_sm100.o: fprint_mat.h mma_tma_sm100.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" mma_tma_sm100.cu

clean:
	rm -rvf *.o *.ii *.cubin *.cudafe1.* *.fatbin* *.reg.c *.module_id *.ptx mma_tma_sm100
