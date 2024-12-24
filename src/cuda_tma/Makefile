CUDA_INCLUDE = /usr/local/cuda/include
CXX=nvcc
CXXFLAGS=-keep -std=c++17 -g -O0 --generate-code=arch=compute_90a,code=sm_90a -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing
CXXFLAGS+= --expt-relaxed-constexpr -D__CUDA_ARCH_FEAT_SM90_ALL
CXXFLAGS+= -I$(CUDA_INCLUDE)

LDLIBS=-lcuda


.SUFFIXES: .o .cu

default: clean main2

main1: main1.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main1.o $(LDLIBS)

main1.o: main1.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" main1.cu

main2: main2.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main2.o $(LDLIBS)

main2.o: main2.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" main2.cu

clean: 
	rm -f *.o output.txt main1 main2
