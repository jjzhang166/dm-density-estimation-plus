#ifndef __TRICONVERTER__
#define __TRICONVERTER__
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "types.h"
#include "tetracut.h"
#include "tetrahedron.h"
#include "triangle.h"
#include "triheader.h"

using namespace std;
//render the tetrahedrons into triangle file
//those triangles are 2D-triangle with velocity values
class TriConverter{
public:
    
            TriConverter        (
                                  int imagesize,
                                  float boxsize,
                                  string prefix,
                                  string outputbasename,
                                  int outputBufferSize = 512
                                );
    
            ~TriConverter        ();
    
    void    process             (Tetrahedron & tetra);
    void    finish              ();    
                                            //the limit of render
                                            //types of this render

private:
    string outputBaseName_;
    string prefix_;
    
    int outputBufferSize_;
    
    Triangle * vertexbuffer_;
    int * vertexIds_;
    int * totalTriangles_;
    
    fstream * outputStreams_;
    
    void    outputPlane(int i);
    
    int     imagesize_;
    int     numplanes_;
    REAL    boxsize_;
    REAL    dz_;
    REAL    startz_;
    
    IsoZCutter cutter;
};

#endif
