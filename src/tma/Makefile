CUDA_INCLUDE = /usr/local/cuda/include

CUTLASS_DIR=../../../cutlass
EXAMPLES_DIR=../
CXX=nvcc
APP=main

CXXFLAGS=--generate-code=arch=compute_90a,code=sm_90a -std=c++17 -O3 -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
CXXFLAGS+= --expt-relaxed-constexpr -D__CUDA_ARCH_FEAT_SM90_ALL
CXXFLAGS+= -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include -I${EXAMPLES_DIR}
CXXFLAGS += -I$(CUDA_INCLUDE)

LDLIBS=-lcuda


.SUFFIXES: .o .cu

default: clean main1

main1: main1.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main1.o $(LDLIBS)

main1.o: main1.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" main1.cu

clean: 
	rm -f *.o main1
