#MACOS
CC=g++
LIB=../library
IFLAG= -I ${LIB} -I ./libkdtree
VELDISPOBJS= velocitydisp.o \
			 readgadget.o tetrahedron.o \
			 triangle.o

VELMEASUREOBJS= velmeasure.o \
			 readgadget.o tetrahedron.o \
			 triangle.o

VELMEASUREPROG= VelMeasure

VELDISPPROG= VelDisp

MACROS=-DTREE_CODE=1

all: obj
	$(CC) $(VELDISPOBJS) -o $(VELDISPPROG) $(LFLAGS)
	$(CC) $(VELMEASUREOBJS) -o $(VELMEASUREPROG) $(LFLAGS)

obj:
	$(CC) -c -Wall $(MACROS) velmeasure.cpp \
			velocitydisp.cpp \
		   ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp \
		   ${LIB}/indtetrahedron.cpp\
		   $(IFLAG)  
omp:
	$(CC) -c -Wall -D__OMP__=1 $(MACROS) velocitydisp.cpp \
		   ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp \
		   ${LIB}/indtetrahedron.cpp\
		   $(IFLAG) -fopenmp
	$(CC) $(VELDISPOBJS) -o $(VELDISPPROG) $(LFLAGS) -fopenmp -lpthread

clean:
	rm -f *.o
	rm -f $(VELDISPPROG) 
	rm -f $(VELMEASUREPROG)

