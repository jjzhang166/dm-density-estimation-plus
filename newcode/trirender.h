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
    
    TriDenRender           (
                            int imagesize,
                            REAL boxSize,
                            string * outputfiles,
                            int numOfOutputs
                           );
    
    static const int maxNumRenderComp;
    

    
    ~TriDenRender          ();
    
    bool good();
    
    //must contains numOfOutputs's outputs
    //floatPerVerts contains 1 or 3
    /*void    rend        (string vertexfile,
                         string * componentFiles,
                         int * floatPerTriangle
                         );*/
    
    void    rend        (float * vertexdata,
                         float * densitydata,
                         int numtriangles
                         );
    
    void close();
    
private:
    int numOfOutputs_;
    
    void init();
    
    /*void setOutputFile(
                       string * outputfiles,
                       int numOfOutputs
                       );*/
    
    int     imagesize_;
    REAL    boxsize_;

    float*  result_;
    
    //the four component of the output color
    fstream * outputStream_;

};

#endif
