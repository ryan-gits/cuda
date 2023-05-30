all:
	nvcc hello.cu -std=c++17 --expt-relaxed-constexpr

run:
	./a.out lenna.bmp 13

prof:
	nvprof ./a.out lenna.bmp 13