CUDA_INCLUDE = /usr/local/cuda/include

CXX_FLAGS = -std=c++17


# all: main
# 	./main

main: main.cc
	$(CXX) $(CXX_FLAGS) -I$(CUDA_INCLUDE) -o main main.cc

clean:
	rm -rvf *.o main
