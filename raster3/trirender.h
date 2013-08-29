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
    
    TriRender            (
                          int imagesize,
                          string outputfile,
                          REAL boxSize
                        );
    
    ~TriRender          ();
    
    void    rend        (string planeFile);
    
private:
    
    int     imagesize_;
    REAL    boxsize_;

    float*  result_;
    
    fstream outputStream_;

    float   startz_;
    float   dz_;
};

#endif
