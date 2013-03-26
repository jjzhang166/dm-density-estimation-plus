#MACOS
CC=g++
LIB=../library
LFLAGS=-framework Carbon -framework OpenGL -framework GLUT 
#LFLAGS= -lGL -lglut -lGLEW
IFLAG= -I /usr/include/GL/ -I${LIB}

all: obj
	$(CC) *.o -o renddens $(LFLAGS)
	@echo success

obj:
	$(CC) -c -Wall main.cpp buffers.cpp isoplane.cpp ${LIB}\processbar.cpp render.cpp\
		   tetrastream.cpp isoplancut.cpp ${LIB}\readgadget.cpp ${LIB}\tetrahedron.cpp triangle.cpp\
		   $(IFLAG)  

clean:
	rm -f *.o
	rm -f renddens 
