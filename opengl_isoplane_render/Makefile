#MACOS
CC=g++

LFLAGS=-framework Carbon -framework OpenGL -framework GLUT 
#LFLAGS= -lGL -lglut -lGLEW
IFLAG= -I /usr/include/GL/

all: obj
	$(CC) *.o -o renddens $(LFLAGS)
	@echo success

obj:
	$(CC) -c -Wall main.cpp buffers.cpp isoplane.cpp processbar.cpp render.cpp\
		   tetrastream.cpp isoplancut.cpp readgadget.cpp tetrahedron.cpp triangle.cpp\
		   $(IFLAG)  

clean:
	rm -f *.o
	rm -f renddens 
