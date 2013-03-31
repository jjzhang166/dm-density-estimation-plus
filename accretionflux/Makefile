#MACOS
CC=g++
LIB=../library
LFLAGS=
IFLAG= -I /usr/include/GL/ -I${LIB}

all: obj
	$(CC) *.o -o accretionrate $(LFLAGS)
	@echo success

obj:
	$(CC) -c -Wall -G main.cpp accretion.cpp ${LIB}/processbar.cpp\
		   ${LIB}/tetrastream.cpp ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp ${LIB}/tetracut.cpp\
		   ${LIB}/indtetrastream.cpp ${LIB}/indtetrahedron.cpp  $(IFLAG)  

clean:
	rm -f *.o
	rm -f accretionrate 
