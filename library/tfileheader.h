#ifndef __LYTFILEHEADER__
#define __LYTFILEHEADER__

#if defined(_WIN32) || defined(_WIN64)
#include <integers>
#endif


//a file contains only tetrahedrons


class TFileHeader{
public:
    uint64_t numOfTetrahedrons;
    float boxSize;
    int other[60];
};

#endif

