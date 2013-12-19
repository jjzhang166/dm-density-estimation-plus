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
    
    static const int VELX;
    static const int VELY;
    static const int VELZ;
    static const int DENS;
    static const int POS;
    
            TriConverter        ( int imagesize,
                                  float boxsize,
                                  string outputbase,
                                  int outputBufferSize = 1024*1024
                                );
    
            ~TriConverter        ();
    
    void    setOutput           (int ouputCode); //connected by "|"
    
    void    process             (Tetrahedron & tetra);
    void    finish              ();    
                                            //the limit of render
                                            //types of this render

private:
    
    void writeToFile(int type,
                     int planeId,
                     ios_base::openmode mode,
                     const char* s,
                     streamsize n,
                     bool isHeader = false
                     );
    
    string outputBaseName_;
    string prefix_;
    
    int outputBufferSize_;
    
    float * vertexbuffer_;
    float * densbuffer_;
    float * velXbuffer_;
    float * velYbuffer_;
    float * velZbuffer_;
    //float * velocitybuffer_;
    int * vertexIds_;
    int * totalTriangles_;
    
    //fstream * outputStreams_;
    
    void    outputPlane(int i);
    
    int     imagesize_;
    int     numplanes_;
    REAL    boxsize_;
    REAL    dz_;
    REAL    startz_;
    
    IsoZCutter cutter;
    
    bool isVelX_;
    bool isVelY_;
    bool isVelZ_;
    bool isPosition_;
    bool isDensity_;
};

#endif
