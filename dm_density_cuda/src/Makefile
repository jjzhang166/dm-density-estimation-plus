NVCCSRC=kernel.cu estimator.cpp
GPPSRC=readgadget.cpp processbar.cpp main.cpp tetrastream.cpp

all:
	nvcc -c ${NVCCSRC}
	g++ -c -Wall ${GPPSRC}
	nvcc -o ../bin/dmdens *.o

clean:
	rm -f *.o
	rm -fr ../bin
