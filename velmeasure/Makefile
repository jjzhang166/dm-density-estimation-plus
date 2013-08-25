#MACOS
CC=g++
LIB=../library
IFLAG= -I ${LIB} -I ./libkdtree
VELDISPOBJS= velocitydisp.o \
			 readgadget.o tetrahedron.o \
			 triangle.o
VELDISPPROG= VelDisp

all: obj
	$(CC) $(VELDISPOBJS) -o $(VELDISPPROG) $(LFLAGS)

obj:
	$(CC) -c -Wall velocitydisp.cpp ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp \
		   ${LIB}/indtetrahedron.cpp\
		   $(IFLAG)  

clean:
	rm -f *.o
	rm -f $(VELDISPPROG) 

