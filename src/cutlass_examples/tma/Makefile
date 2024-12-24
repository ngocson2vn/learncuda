CUDA_INCLUDE = /usr/local/cuda/include

CUTLASS_DIR=../../../cutlass
EXAMPLES_DIR=../
CXX=nvcc
APP=main

# CXXFLAGS=-g -O0 -std=c++17 --device-debug
CXXFLAGS=-std=c++17 -O3
# CXXFLAGS+= -Xcompiler=-fno-inline
CXXFLAGS+= --generate-code=arch=compute_90a,code=sm_90a
CXXFLAGS+= --expt-relaxed-constexpr -D__CUDA_ARCH_FEAT_SM90_ALL
# CXXFLAGS+= -DCUTLASS_ENABLE_SYNCLOG
CXXFLAGS+= -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include -I${EXAMPLES_DIR}
CXXFLAGS+= -I$(CUDA_INCLUDE)

INCLUDE_DIR=-I$(CUDA_INCLUDE) -I${CUTLASS_DIR}/include -I${CUTLASS_DIR}/examples/common -I${CUTLASS_DIR}/tools/util/include -I${EXAMPLES_DIR}
HOST_CXX_FLAGS = -std=c++17 -ggdb -O0 -fno-inline -fno-limit-debug-info -DCUTE_COPY_ATOM_TMA_SM90_ENABLED -D__CUDA_ARCH_FEAT_SM90_ALL
LDLIBS=-lcuda


.SUFFIXES: .o .cu

default: clean main

main: main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main.o $(LDLIBS)

main.o: main.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" main.cu

test: test.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ test.o $(LDLIBS)

test.o: test.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" test.cu

test2: test2.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ test2.o $(LDLIBS)

test2.o: test2.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" test2.cu

dump: test.cc
	clang++ $(HOST_CXX_FLAGS) -Xclang -ast-print -fsyntax-only $(INCLUDE_DIR) test.cc > test.cxx

clean: 
	rm -f *.o test $(APP)
