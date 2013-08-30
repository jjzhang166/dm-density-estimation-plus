#MACOS
CC=g++
LIB=../library
IFLAG= -I ${LIB}
CONVERTOBJS= tconverter.o indtetrastream.o \
			 readgadget.o tetrahedron.o \
			 indtetrahedron.o tfilestream.o\
			 triangle.o tetrastream.o
CONVERTNAME= TFileConverter

TRISLICEPROG=TriSlicer
TRISLICEOBJ=trislice.o triconverter.o	\
			indtetrastream.o	\
		   	readgadget.o tetrastream.o	\
		   	tetrahedron.o triangle.o indtetrahedron.o\
		   	tetracut.o

all: obj
	$(CC) $(CONVERTOBJS) -o $(CONVERTNAME) $(LFLAGS)
	$(CC) $(TRISLICEOBJ) -o $(TRISLICEPROG) $(LFLAGS)

obj:
	$(CC) -c -Wall tconverter.cpp ${LIB}/processbar.cpp\
		   ${LIB}/tetrastream.cpp ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp \
		   ${LIB}/tetracut.cpp ${LIB}/tfilestream.cpp\
		   ${LIB}/indtetrastream.cpp ${LIB}/indtetrahedron.cpp\
		   triconverter.cpp trislice.cpp \
		   $(IFLAG)  

clean:
	rm -f *.o
	rm -f $(CONVERTNAME) 
	rm -f $(TRISLICEPROG)
