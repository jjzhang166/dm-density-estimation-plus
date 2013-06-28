#ifndef __DENRENDER__
#define __DENRENDER__
#include "types.h"
#include "buffers.h"
#include "tetracut.h"
#include "tetrahedron.h"

//render the density through a serier plane cutter of the tetrahedrons
//startz, dz, and numplane specifies how many slices of the density field
//will be rendered
class DenRender{
public:
    DenRender(int imagesize, float boxsize,
              float startz, float dz, int numplane,
              int * argc, char ** args);
    
    ~DenRender();
    
    void rend(Tetrahedron & tetra);
    
    float * getDenfield();
    
    float * getImage();
    fluxBuffer ** getBuffers();
    
private:
    void openGLInit();
    
    int * argc_;
    char ** args_;
    
    int imagesize_;
    REAL boxsize_;
    REAL viewSize;
    int numplanes_;
    
    float * image_;
    float startz_, dz_;
    
    fluxBuffer **fbuffer;

    IsoZCutter cutter;
};

#endif
