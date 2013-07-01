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
    void finish();
    
    float * getDenfield();
    
    float * getImage();
    
    
    static const int VERTEXBUFFERDEPTH;
    

private:
    
    void rendplane(int i);
    
    void openGLInit();
    
    int * argc_;
    char ** args_;
    
    int imagesize_;
    REAL boxsize_;
    REAL viewSize;
    int numplanes_;
    
    //image stores all the density field
    float * image_;
    //image stores only one slides of the field
    float * tempimage_;
    float startz_, dz_;
    
    buffer *fbuffer;
    //the buffer for drawing triangles
    float * vertexbuffer_;
    int * vertexIds_;
    
    IsoZCutter cutter;
};

#endif
