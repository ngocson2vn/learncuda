NVCC = /usr/local/cuda/bin/nvcc

main: ws.o main.o
	g++ -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -o main main.o ws.o -lcudart

main.o: main.cc
	g++ -O0 -g -I/usr/local/cuda/include -c main.cc

ws.o: ws.h ws.cu
	$(NVCC) -gencode arch=compute_90,code=sm_90 -O0 -g -I/usr/local/cuda/include -c ws.cu

clean:
	rm -rvf *.o main
