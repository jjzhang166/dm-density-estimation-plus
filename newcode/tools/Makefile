CC=g++
LFLAG= -framework Carbon -framework OpenGL -framework GLUT
#LFLAG=-lGL -lglut -lGLEW
#LFLAG= -lGL -lglut /home/lyang/Tools/glew/lib/libGLEW.a
#IFLAG= -I./
IFLAG= -I /usr/include/GL/ -I./ -I ~/Tools/glew/include/ -I ../GadgetReader/

SOURCESFILES=../GadgetReader/gadgetreader.cpp \
			 ../GadgetReader/read_utils.c\
 			 densprofile.cpp			 

CPPOBJS= densprofile.o gadgetreader.o read_utils.o \

DENSPROF= DensProfile

DENSPROFOBJS= densprofile.o gadgetreader.o read_utils.o

all: $(DENSPROF) 

$(DENSPROF) densprof: $(DENSPROFOBJS)
	$(CC) -o $(DENSPROF) $(DENSPROFOBJS)

$(CPPOBJS): $(SOURCESFILES)
	$(CC) -Wall -c $(SOURCESFILES) $(IFLAG)

clean:
	rm -fr *.o
	rm -fr $(DENSPROF)


