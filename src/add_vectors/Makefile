NVCC = /usr/local/cuda/bin/nvcc

main: matSumKernel.ptx main.cpp
	$(CXX) -O0 -g -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcuda -o main main.cpp 

matSumKernel.ptx: matSumKernel.h matSumKernel.cu
	$(NVCC) -O0 -g --ptx -c matSumKernel.cu

clean:
	rm -rvf *.o *.ptx main