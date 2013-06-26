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

MELTFEOBJ=  tetrastream.o readgadget.o accretion.o processbar.o\
		tetrahedron.o triangle.o tetracut.o haloread.o\
		indtetrastream.o indtetrahedron.o measureacc_ltfe.o

MASKOBJ= tetrastream.o readgadget.o accretion.o processbar.o\
		tetrahedron.o triangle.o tetracut.o haloread.o\
		indtetrastream.o indtetrahedron.o maskprod.o


all: obj
	$(CC) $(ACCOBJ) -o accretionrate $(LFLAGS)
	$(CC) $(HPOBJ) -o haloprop
	$(CC) $(MELTFEOBJ) -o measureacc_ltfe $(LFLAGS)  $(FITSIOOBJ)
	$(CC) $(MASKOBJ) -o maskprod $(LFLAGS)  $(FITSIOOBJ)
	@echo success

obj:
	$(CC) -c -Wall main.cpp accretion.cpp ${LIB}/processbar.cpp\
		   ${LIB}/tetrastream.cpp ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp ${LIB}/tetracut.cpp\
		   ${LIB}/indtetrastream.cpp ${LIB}/indtetrahedron.cpp\
		   measureacc_ltfe.cpp $(IFLAG)  -I $(FITSIODIR)
	$(CC) -c haloprop.cpp -I $(FITSIODIR)
	$(CC) -c haloread.cpp -I $(FITSIODIR)
	$(CC) -c maskprod.cpp -I $(FITSIODIR) $(IFLAG) 


test:
	$(CC) -c testhaloread.cpp haloread.cpp -I $(FITSIODIR)
	$(CC) -o testhaloread testhaloread.o haloread.o $(FITSIOOBJ)


clean:
	rm -f *.o
	rm -f accretionrate 
	rm -f haloprop
	rm -f testhaloread
	rm -f measureacc_ltfe
	rm -f maskprod
