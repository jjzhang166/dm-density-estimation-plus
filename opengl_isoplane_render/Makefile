#MACOS
CC=g++
LIB=../library
LFLAGS=-framework Carbon -framework OpenGL -framework GLUT 
#LFLAGS= -lGL -lglut -lGLEW
IFLAG= -I /usr/include/GL/ -I${LIB}


RENDOBJ=main.o buffers.o isoplane.o processbar.o render.o\
		   indtetrastream.o isoplancut.o readgadget.o \
		   tetrahedron.o triangle.o tetracut.o \
		   tetrastream.o indtetrahedron.o

TOBJ=testisozcutter.o buffers.o isoplane.o processbar.o render.o\
		   indtetrastream.o isoplancut.o readgadget.o \
		   tetrahedron.o triangle.o tetracut.o \
		   tetrastream.o indtetrahedron.o

TDROBJ=testdensrender.o buffers.o processbar.o\
		   indtetrastream.o readgadget.o tetrastream.o\
		   tetrahedron.o triangle.o indtetrahedron.o\
		   denrender.o tetracut.o

ZDENSOBJ=cutzdens.o buffers.o processbar.o\
		   indtetrastream.o readgadget.o tetrastream.o\
		   tetrahedron.o triangle.o indtetrahedron.o\
		   denrender.o tetracut.o

DENSVIEWOBJ=viewfun.o densviewer.o


all: obj
	$(CC) $(RENDOBJ) -o renddens $(LFLAGS)
	$(CC) $(TOBJ) -o testisoz $(LFLAGS)
	$(CC) $(TDROBJ) -o testdender $(LFLAGS)
	$(CC) $(ZDENSOBJ) -o zdensrender $(LFLAGS)
	$(CC) $(DENSVIEWOBJ) -o DensViewer $(LFLAGS)
	@echo success

obj:
	$(CC) -c -Wall main.cpp buffers.cpp isoplane.cpp ${LIB}/processbar.cpp render.cpp\
		   ${LIB}/indtetrastream.cpp isoplancut.cpp ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp ${LIB}/tetracut.cpp \
		   ${LIB}/tetrastream.cpp ${LIB}/indtetrahedron.cpp testisozcutter.cpp\
		   testdensrender.cpp denrender.cpp cutzdens.cpp viewfun.cpp densviewer.cpp\
 			$(IFLAG)  

clean:
	rm -f *.o
	rm -f renddens 
	rm -f testisoz
	rm -f testdender
	rm -f zdensrender
	rm -f DensViewer
