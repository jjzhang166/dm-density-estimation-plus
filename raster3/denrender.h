#ifndef __DENRENDER__
#define __DENRENDER__
#include "types.h"
//#include "buffers.h"
#include "tetracut.h"
#include "tetrahedron.h"

//how many color components are being calculated
#define NUMCOLORCOMP 2

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
    
    int * getStreamData();
    
    
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
    
    //stores the stream data of any pixels
    int * streams_;
    
    float * density_;
    
    //image stores only one slides of the field
    float startz_, dz_;
    
    //buffer *fbuffer;
    //the buffer for drawing triangles
    float * vertexbuffer_;
    int * vertexIds_;
    
    IsoZCutter cutter;
};

#endif
