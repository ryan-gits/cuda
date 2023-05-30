all:
	mkdir build
	nvcc ./src/hello.cu -std=c++17 --expt-relaxed-constexpr -I ./src -o ./build/hello

run:
	./build/hello ./resources/lenna.bmp 13

prof:
	nvprof ./build/hello ./resources/lenna.bmp 13

clean:
	rm -rf ./build