CUDA_INCLUDE = /usr/local/cuda-12.2/include
CUTLASS_INCLUDE = ../../../cutlass/include
CUTLASS_UTIL_INCLUDE = ../../../cutlass/tools/util/include
CXX=/usr/local/cuda/bin/nvcc
CXXFLAGS=-std=c++17 -g -O0

# Suppress note: The ABI for passing parameters with 64-byte alignment has changed in GCC 4.6
CXXFLAGS+= -Xcompiler "-Wno-psabi -fno-inline-functions"
CXXFLAGS+= -lineinfo
CXXFLAGS+= --generate-code=arch=compute_90a,code=sm_90a
CXXFLAGS+= --expt-relaxed-constexpr -D__CUDA_ARCH_FEAT_SM90_ALL
CXXFLAGS+= -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL_INCLUDE)
ALL_INCLUDE = -I$(CUDA_INCLUDE) -I$(CUTLASS_INCLUDE) -I$(CUTLASS_UTIL_INCLUDE)

LIB_DIRS=-L/usr/local/cuda-12.2/lib64
LDLIBS=-lcudart -lcuda

.SUFFIXES: .o .cu

default: clean wgmma_tma_sm90_debug

wgmma_tma_sm90: clean wgmma_tma_sm90.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIB_DIRS) -o $@ wgmma_tma_sm90.o $(LDLIBS)

wgmma_tma_sm90.o: fprint_mat.h wgmma_tma_sm90.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" wgmma_tma_sm90.cu

wgmma_tma_sm90_debug: clean wgmma_tma_sm90_debug.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(LIB_DIRS) -o $@ wgmma_tma_sm90_debug.o $(LDLIBS)

wgmma_tma_sm90_debug.o: fprint_mat.h wgmma_tma_sm90_debug.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" wgmma_tma_sm90_debug.cu

test: clean_test
	clang++ -std=c++17 -g -O0 -fno-inline $(ALL_INCLUDE) -o test test.cpp

clean:
	rm -rvf *.o main test wgmma_tma_sm90 wgmma_tma_sm90_debug

clean_test:
	rm -rvf test.o test