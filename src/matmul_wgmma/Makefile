CUDA_INCLUDE = /usr/local/cuda/include
CXX=/usr/local/cuda/bin/nvcc
CXXFLAGS=-std=c++17 -g -O0

# Suppress note: The ABI for passing parameters with 64-byte alignment has changed in GCC 4.6
CXXFLAGS+= -Xcompiler -Wno-psabi

CXXFLAGS+= --generate-code=arch=compute_90a,code=sm_90a
CXXFLAGS+= --expt-relaxed-constexpr -D__CUDA_ARCH_FEAT_SM90_ALL
CXXFLAGS+= -I$(CUDA_INCLUDE)

LDLIBS=-lcuda

.SUFFIXES: .o .cu

default: clean main

main: main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main.o $(LDLIBS)

main.o: device.h fprint_mat.h main.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" main.cu

test: fprint_mat.h test.cu
	$(NVCC) $(NVCC_FLAGS) -I$(CUDA_INCLUDE) -o test test.cu

clean:
	rm -rvf *.o main
