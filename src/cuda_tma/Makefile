CUDA_INCLUDE = /usr/local/cuda/include
CXX=/usr/local/cuda/bin/nvcc
CXXFLAGS=-std=c++17 -g -O0 -Xcompiler -Wno-psabi --generate-code=arch=compute_90a,code=sm_90a
CXXFLAGS+= --expt-relaxed-constexpr -D__CUDA_ARCH_FEAT_SM90_ALL
CXXFLAGS+= -I$(CUDA_INCLUDE)

LDLIBS=-lcuda


.SUFFIXES: .o .cu

default: clean main3

main1: main1.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main1.o $(LDLIBS)

main1.o: main1.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" main1.cu

main2: main2.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main2.o $(LDLIBS)

main2.o: main2.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" main2.cu

main2.ptx: main2.cu
	$(CXX) $(CXXFLAGS) -ptx -o "$@" main2.cu

main3: main3.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) -o $@ main3.o $(LDLIBS)

main3.o: main3.cu
	$(CXX) -c $(CXXFLAGS) -o "$@" main3.cu

clean: 
	rm -fv *.o output.txt main1 main2 main3
