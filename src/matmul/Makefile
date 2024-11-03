NVCC = /usr/local/cuda/bin/nvcc

main: matmul.o main.o
	g++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o main main.o matmul.o -lcudart

main.o: main.cc
	$(NVCC) -O0 -g -I/usr/local/cuda/include -c main.cc

matmul.o: matmul.h matmul.cu
	$(NVCC) -O0 -g -I/usr/local/cuda/include -c matmul.cu

clean:
	rm -rvf *.o main
