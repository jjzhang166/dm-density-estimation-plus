#ifndef __TRIRENDER__
#define __TRIRENDER__
#include <vector>
#include <string>
#include "types.h"
//#include "buffers.h"
#include "tetracut.h"
#include "tetrahedron.h"

//render the density through a serier plane cutter of the tetrahedrons
//startz, dz, and numplane specifies how many slices of the density field
//will be rendered

//render density
class TriDenRender{
public:
    
    TriDenRender            (
                            int imagesize,
                            REAL boxSize
                            );
    
    static const int maxNumRenderComp;
    

    
    ~TriDenRender          ();
    
    

    
    void rendDensity    (float * vertexdata,
                         float * densitydata,
                         int numtriangles,
                         bool isClear = true
                         );
    
    // render the velocity filed
    // TODO
    void rendDensity    (float * vertexdata,
                         float * densitydata,
                         float * velxdata,
                         float * velydata,
                         float * velzdata,
                         int numtriangles,
                         bool isClear = true
                         );
    
    float * getDensity();
    float * getVelocityX();
    float * getVelocityY();
    float * getVelocityZ();
    
private:
    //int numOfOutputs_;
    
    void init();
    
    int     imagesize_;
    REAL    boxsize_;

    //float*  result_;
    
    
    float * density_;
    float * velocityx_;
    float * velocityy_;
    float * velocityz_;
    
    float * colorData;
    
    
    void rend(int NumTriangles, float * vertexdata);
    //the four component of the output color
    //fstream * outputStream_;

};

#endif
