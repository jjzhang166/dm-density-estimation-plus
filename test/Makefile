#MACOS
CC=g++
LIB=../library
#LFLAGS=-framework Carbon -framework OpenGL -framework GLUT 
#LFLAGS= -lGL -lglut -lGLEW
IFLAG= -I /usr/include/GL/ -I${LIB}

all: obj
	$(CC) testtetrastreamer.o indtetrastream.o readgadget.o \
			tetracut.o triangle.o tetrastream.o tetrahedron.o\
		   	-o testtetrastreamer $(LFLAGS)
	@echo success

obj:
	$(CC) -c -G -Wall ${LIB}/processbar.cpp\
			${LIB}/indtetrastream.cpp ${LIB}/readgadget.cpp \
		   	${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp ${LIB}/tetracut.cpp \
		   	${LIB}/tetrastream.cpp testtetrastreamer.cpp $(IFLAG)  

clean:
	rm -f *.o
	rm -f testtetrastreamer 
