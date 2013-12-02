#MACOS
CC=g++
CUDACC=nvcc
LIB=../library
#LFLAGS= -framework Carbon -framework OpenGL -framework GLUT 
#LFLAGS= -lGL -lglut -lGLEW
LFLAGS= -lGL -lglut /home/lyang/Tools/glew/lib/libGLEW.a
#IFLAG= -I /usr/include/GL/ -I${LIB}
IFLAG= -I /usr/include/GL/ -I${LIB} -I ~/Tools/glew/include/
PROFILEFLAGS= -g


RENDPROG=renddens_1slide
RENDOBJ=main.o buffers.o isoplane.o processbar.o render.o\
		   indtetrastream.o isoplancut.o readgadget.o \
		   tetrahedron.o triangle.o tetracut.o \
		   tetrastream.o indtetrahedron.o

TRIDENSPROG=TriDens
TRIDENSOBJ=tridens.o trirender.o buffers.o processbar.o\
		   indtetrastream.o readgadget.o tetrastream.o\
		   tetrahedron.o triangle.o indtetrahedron.o

TPROG=testisoz
TOBJ=testisozcutter.o buffers.o isoplane.o processbar.o render.o\
		   indtetrastream.o isoplancut.o readgadget.o \
		   tetrahedron.o triangle.o tetracut.o \
		   tetrastream.o indtetrahedron.o
TDRPROG=testdender
TDROBJ=testdensrender.o buffers.o processbar.o\
		   indtetrastream.o readgadget.o tetrastream.o\
		   tetrahedron.o triangle.o indtetrahedron.o\
		   denrender.o tetracut.o

ZDENSPROG=CutZRender
ZDENSOBJ=cutzdens.o buffers.o processbar.o\
		   indtetrastream.o readgadget.o tetrastream.o\
		   tetrahedron.o triangle.o indtetrahedron.o\
		   denrender.o tetracut.o

ZDENSPROG=CutZRender
ZDENSOBJ=cutzdens.o buffers.o processbar.o\
		   indtetrastream.o readgadget.o tetrastream.o\
		   tetrahedron.o triangle.o indtetrahedron.o\
		   denrender.o tetracut.o


TFILEPROG=TFileZrender
TFILEOBJ=tfile_render.o buffers.o processbar.o\
		   indtetrastream.o readgadget.o tetrastream.o\
		   tetrahedron.o triangle.o indtetrahedron.o\
		   denrender.o tetracut.o tfilestream.o


CUDAZDENSPROG=cudaZDensRender
CUDAZDENSOBJ=cutzdens.o processbar.o\
			 indtetrastream.o readgadget.o tetrastream.o\
			 cudakernel.o indtetrahedron.o cudarender.o\
			 tetracut.o

DENSVIEWPROG=DensViewer
DENSVIEWOBJ=viewfun.o densviewer.o


all: obj
	$(CC) $(RENDOBJ) -o $(RENDPROG) $(LFLAGS)
	$(CC) $(TOBJ) -o $(TPROG) $(LFLAGS)
	$(CC) $(TDROBJ) -o $(TDRPROG) $(LFLAGS)
	$(CC) $(ZDENSOBJ) -o $(ZDENSPROG) $(LFLAGS)
	$(CC) $(DENSVIEWOBJ) -o $(DENSVIEWPROG) $(LFLAGS)
	$(CC) $(TFILEOBJ) -o $(TFILEPROG) $(LFLAGS)
	$(CC) $(TRIDENSOBJ) -o $(TRIDENSPROG) $(LFLAGS)
	@echo success

obj:
	$(CC) -c -Wall main.cpp buffers.cpp isoplane.cpp ${LIB}/processbar.cpp render.cpp\
		   ${LIB}/indtetrastream.cpp isoplancut.cpp ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp ${LIB}/tetracut.cpp \
		   ${LIB}/tetrastream.cpp ${LIB}/indtetrahedron.cpp testisozcutter.cpp\
		   testdensrender.cpp denrender.cpp cutzdens.cpp viewfun.cpp densviewer.cpp\
 		   tfile_render.cpp ${LIB}/tfilestream.cpp \
		   tridens.cpp trirender.cpp $(IFLAG)  


profile:
	$(CC) $(PROFILEFLAGS) -c -Wall main.cpp buffers.cpp isoplane.cpp ${LIB}/processbar.cpp render.cpp\
		   ${LIB}/indtetrastream.cpp isoplancut.cpp ${LIB}/readgadget.cpp \
		   ${LIB}/tetrahedron.cpp ${LIB}/triangle.cpp ${LIB}/tetracut.cpp \
		   ${LIB}/tetrastream.cpp ${LIB}/indtetrahedron.cpp testisozcutter.cpp\
		   testdensrender.cpp denrender.cpp cutzdens.cpp viewfun.cpp densviewer.cpp\
 		   tfile_render.cpp ${LIB}/tfilestream.cpp \
		   tridens.cpp trirender.cpp $(IFLAG)  
	$(CC) $(PROFILEFLAGS) $(RENDOBJ) -o $(RENDPROG) $(LFLAGS)
	$(CC) $(PROFILEFLAGS) $(TOBJ) -o $(TPROG) $(LFLAGS)
	$(CC) $(PROFILEFLAGS) $(TDROBJ) -o $(TDRPROG) $(LFLAGS)
	$(CC) $(PROFILEFLAGS) $(ZDENSOBJ) -o $(ZDENSPROG) $(LFLAGS)
	$(CC) $(PROFILEFLAGS) $(DENSVIEWOBJ) -o $(DENSVIEWPROG) $(LFLAGS)
	$(CC) $(PROFILEFLAGS) $(TFILEOBJ) -o $(TFILEPROG) $(LFLAGS)
	$(CC) $(PROFILEFLAGS) $(TRIDENSOBJ) -o $(TRIDENSPROG) $(LFLAGS)
	@echo success

cuda:
	$(CC) -c -Wall ${LIB}/processbar.cpp\
			${LIB}/indtetrastream.cpp ${LIB}/readgadget.cpp \
			${LIB}/tetracut.cpp ${LIB}/tetrastream.cpp ${LIB}/indtetrahedron.cpp \
			cutzdens.cpp $(IFLAG)  

	$(CUDACC) -c cudarender.cpp cudakernel.cu $(IFLAG)
	$(CUDACC) -o $(CUDAZDENSPROG) $(CUDAZDENSOBJ) 	

clean:
	rm -f *.o
	rm -f $(RENDPROG)  
	rm -f $(TPROG)
	rm -f $(TDRPROG)
	rm -f $(ZDENSPROG)
	rm -f $(DENSVIEWPROG)
	rm -f $(CUDAZDENSPROG)
	rm -f $(TRIDENSPROG)
	rm -f $(TFILEPROG)
