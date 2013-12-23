CC=g++
#LFLAG= -framework Carbon -framework OpenGL -framework GLUT
LFLAG=-lGL -lglut -lGLEW
#LFLAGS= -lGL -lglut /home/lyang/Tools/glew/lib/libGLEW.a
IFLAG=-I /usr/include/GL/ -I./
#IFLAG= -I /usr/include/GL/ -I./ -I ~/Tools/glew/include/

SOURCESFILES=filedivider.cpp GadgetReader/gadgetreader.cpp \
			 GadgetReader/read_utils.c io_utils.cpp\
			 processbar.cpp test.cpp \
			 tetrahedron.cpp indtetrahedron.cpp \
			 dtetrastream.cpp trislicer.cpp\
			 triangle.cpp trirender.cpp\
			 tetracut.cpp triconverter.cpp \
			 tridens.cpp buffers.cpp densviewer.cpp\
			 viewfun.cpp trifile_util.cpp

FILEDIVIER=FileDivider
TRISLICER=TriSlicer
TRIRENS=TriDensRender
DENSVIEWER=DensViewer


all: objs
	$(CC) -o $(FILEDIVIER) filedivider.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o
	$(CC) -o $(TRISLICER) trislicer.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o\
			tetrahedron.o indtetrahedron.o dtetrastream.o \
			triconverter.o tetracut.o triangle.o trifile_util.o
	$(CC) -o $(TRIRENS) tridens.o io_utils.o \
			processbar.o gadgetreader.o read_utils.o\
			tetrahedron.o indtetrahedron.o dtetrastream.o \
			triconverter.o tetracut.o triangle.o trifile_util.o\
			buffers.o trirender.o $(LFLAG)
	$(CC) -o $(DENSVIEWER) densviewer.o viewfun.o $(LFLAG)


objs:
	$(CC) -Wall -c $(SOURCESFILES) $(IFLAG)

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
	rm -fr *.o
	rm -fr $(FILEDIVIER)
	rm -fr $(TRISLICER)
	rm -fr $(TRIRENS)
	rm -fr $(DENSVIEWER)
	rm -fr readtest


