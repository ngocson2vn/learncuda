NVCC = /usr/local/cuda/bin/nvcc

main: main.cpp cuda.o
	$(CXX) -O0 -g -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -o main cuda.o main.cpp 

main2: main2.cpp cuda.o
	$(CXX) -O0 -g -I/usr/local/cuda/include -L/usr/local/cuda/lib64 -lcudart -o main2 cuda.o main2.cpp

cuda.o: cuda.h cuda.cu
	$(NVCC) -O0 -g -c cuda.cu

clean:
	rm -rvf *.o main main2