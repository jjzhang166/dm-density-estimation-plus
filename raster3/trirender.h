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
    
    TriDenRender        (
                          int imagesize,
                          string outputfile,
                          REAL boxSize
                        );
    
    ~TriDenRender          ();
    
    bool good();
    
    void    rend        (string trifile, string denfile);
    
    void close();
    
private:
    
    void init();
    
    int     imagesize_;
    REAL    boxsize_;

    float*  result_;
    
    fstream outputStream_;
};

#endif
