#CC=g++
CC=g++ -std=c++0x
#LFLAG= -framework Carbon -framework OpenGL -framework GLUT
LFLAG=-lGL -lglut -lGLEW
#LFLAG= -lGL -lglut /home/lyang/Tools/glew/lib/libGLEW.a
#IFLAG= -I./
IFLAG= -I /usr/include/GL/ -I./ #-I ~/Tools/glew/include/

SOURCESFILES= filedivider.cpp GadgetReader/gadgetreader.cpp \
			 GadgetReader/read_utils.c io_utils.cpp\
			 processbar.cpp \
			 tetrahedron.cpp indtetrahedron.cpp \
			 dtetrastream.cpp trislicer.cpp\
			 triangle.cpp tetracut.cpp triconverter.cpp \
			 viewfun.cpp trifile_util.cpp cic.cpp \
			 cicdens.cpp

CPPOBJS= filedivider.o gadgetreader.o read_utils.o io_utils.o\
		processbar.o tetrahedron.o tetrahedron.o\
		dtetrastream.o trislicer.o triangle.o tetracut.o\
		triconverter.o viewfun.o trifile_util.o\
		cic.o cicdens.o

GLSOURCESFILES=  trirender.cpp buffers.cpp tridens.cpp\
				 densviewer.cpp test.cpp 

GLOBJS= trirender.o buffers.o tridens.o densviewer.o test.o 


FILEDIVIER=FileDivider
TRISLICER=TriSlicer
TRIRENS=TriDensRender
DENSVIEWER=DensViewer
CICDens=CICDens


FILEDIVIEROBJS= filedivider.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o

TRISLICEROBJS= trislicer.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o\
			tetrahedron.o indtetrahedron.o dtetrastream.o \
			triconverter.o tetracut.o triangle.o trifile_util.o

RENDEROBJS=  tridens.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o\
			tetrahedron.o indtetrahedron.o dtetrastream.o \
			triconverter.o tetracut.o triangle.o trifile_util.o\
			buffers.o trirender.o 

CICOBJS= cic.o cicdens.o gadgetreader.o read_utils.o processbar.o
CICSRC= cic.cpp

VIEWEROBJS= densviewer.o viewfun.o 

all: cic $(FILEDIVIER)  $(TRISLICER) $(TRIRENS) $(DENSVIEWER) 

$(CICDens), cic: $(CICOBJS)
	$(CC) -o CICdens $(CICOBJS)

$(FILEDIVIER) divider: $(FILEDIVIEROBJS)
	$(CC) -o $(FILEDIVIER) $(FILEDIVIEROBJS)

$(TRISLICER) slicer:  $(TRISLICEROBJS)
	$(CC) -o $(TRISLICER) $(TRISLICEROBJS) 

$(TRIRENS) render: $(RENDEROBJS)
	$(CC) -o $(TRIRENS)  $(RENDEROBJS) $(LFLAG)

$(DENSVIEWER) viewer: $(VIEWEROBJS)
	$(CC) -o $(DENSVIEWER) $(VIEWEROBJS) $(LFLAG)


$(CPPOBJS): $(SOURCESFILES)
	$(CC) -Wall -c $(SOURCESFILES) $(IFLAG)

$(GLOBJS): $(GLSOURCESFILES)
	$(CC) -Wall -c $(GLSOURCESFILES) $(IFLAG)
	

debugobjs:
	$(CC) -g -pg -Wall -c $(SOURCESFILES) $(IFLAG)

debug: debugobjs
	$(CC) -pg -g -o $(FILEDIVIER) filedivider.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o
	$(CC) -pg -g -o $(TRISLICER) trislicer.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o\
			tetrahedron.o indtetrahedron.o dtetrastream.o \
			triconverter.o tetracut.o triangle.o trifile_util.o
	$(CC) -g -pg -o $(TRIRENS) tridens.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o\
			tetrahedron.o indtetrahedron.o dtetrastream.o \
			triconverter.o tetracut.o triangle.o trifile_util.o\
			buffers.o trirender.o $(LFLAG)
	$(CC) -g -pg -o $(DENSVIEWER) densviewer.o viewfun.o $(LFLAG)

test: objs
	$(CC) -o readtest test.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o \
			tetrahedron.o indtetrahedron.o dtetrastream.o

clean:
	rm -fr *.o *.so
	rm -fr $(FILEDIVIER)
	rm -fr $(TRISLICER)
	rm -fr $(TRIRENS)
	rm -fr $(DENSVIEWER)
	rm -fr readtest
	rm -fr _cicswigwrapper_module.cc
	rm -fr _cicswigwrapper.so
	rm -fr cicpy.py
	rm -fr cicpy.pyc
	rm -fr _gadgetswigwrapper_module.cc
	rm -fr GADGETPy.py 
