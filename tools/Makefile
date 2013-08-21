#MACOS
CC=g++
LIB=../library
IFLAG= -I ${LIB}
CONVERTOBJS= tconverter.o indtetrastream.o \
			 readgadget.o tetrahedron.o \
			 indtetrahedron.o tfilestream.o\
			 triangle.o tetrastream.o
CONVERTNAME= TFileConverter

all: obj
	$(CC) $(CONVERTOBJS) -o $(CONVERTNAME) $(LFLAGS)

obj:
	$(CC) -c -Wall tconverter.cpp ${LIB}/processbar.cpp\
		   ${LIB}/tetrastream.cpp ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp \
		   ${LIB}/tetracut.cpp ${LIB}/tfilestream.cpp\
		   ${LIB}/indtetrastream.cpp ${LIB}/indtetrahedron.cpp\
		   $(IFLAG)  

clean:
	rm -f *.o
	rm -f $(CONVERTNAME) 

