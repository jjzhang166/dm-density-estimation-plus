#ifndef __LYTFILEHEADER__
#define __LYTFILEHEADER__

//a file contains only tetrahedrons

class TFileHeader{
public:
    uint32_t numOfTetrahedrons;
    float boxSize;
    int other[62];
};

#endif