LIBPATH=../../library
NVCCSRC=kernel.cu estimator.cpp
GPPSRC=${LIBPATH}/readgadget.cpp ${LIBPATH}/processbar.cpp main.cpp tetrastream.cpp

all:
	nvcc -c ${NVCCSRC} -I ${LIBPATH}
	g++ -c -Wall ${GPPSRC} -I ${LIBPATH}
	nvcc -o ../bin/dmdens *.o

debug:
	nvcc -c -g -G ${NVCCSRC} -I ${LIBPATH}
	g++ -c -Wall ${GPPSRC} -I ${LIBPATH}
	nvcc -o ../bin/dmdens-debug *.o

clean:
	rm -f *.o
	rm -fr ../bin
