#ifndef __LYTFILEHEADER__
#define __LYTFILEHEADER__
#include <integers>

//a file contains only tetrahedrons


class TFileHeader{
public:
    uint64_t numOfTetrahedrons;
    float boxSize;
    int other[60];
};

#endif

