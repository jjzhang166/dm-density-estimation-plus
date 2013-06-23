#MACOS
CC=g++
LIB=../library
LFLAGS=
IFLAG= -I /usr/include/GL/ -I${LIB}
FITSIODIR=~/Tools/cfitsio/
FITSIOOBJ=~/Tools/cfitsio/libcfitsio.a

ACCOBJ= tetrastream.o readgadget.o accretion.o processbar.o\
		tetrahedron.o triangle.o tetracut.o\
		indtetrastream.o indtetrahedron.o main.o

HPOBJ= haloprop.o $(FITSIOOBJ)



all: obj
	$(CC) $(ACCOBJ) -o accretionrate $(LFLAGS)
	$(CC) $(HPOBJ) -o haloprop
	@echo success

obj:
	$(CC) -c -Wall -G main.cpp accretion.cpp ${LIB}/processbar.cpp\
		   ${LIB}/tetrastream.cpp ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp ${LIB}/tetracut.cpp\
		   ${LIB}/indtetrastream.cpp ${LIB}/indtetrahedron.cpp  $(IFLAG)  
	$(CC) -c haloprop.cpp -I $(FITSIODIR)

test:
	$(CC) -c testhaloread.cpp haloread.cpp -I $(FITSIODIR)
	$(CC) -o testhaloread testhaloread.o haloread.o $(FITSIOOBJ)


clean:
	rm -f *.o
	rm -f accretionrate 
	rm -f haloprop
	rm -f testhaloread
