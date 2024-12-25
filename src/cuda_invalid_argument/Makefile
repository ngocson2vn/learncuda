CUDA_INCLUDE = /usr/local/cuda/include
CXX=nvcc
CXXFLAGS=-std=c++17 -g -O0 --generate-code=arch=compute_90a,code=sm_90a
CXXFLAGS+= --expt-relaxed-constexpr
CXXFLAGS+= -I$(CUDA_INCLUDE)

LDLIBS=-lcuda

default: clean main

main: main.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main.o $(LDLIBS)

main.o: main.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" main.cu

clean: 
	rm -f *.o main
