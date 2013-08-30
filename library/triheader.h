#ifndef __TRIHEADER__
#define __TRIHEADER__
#include <stdint.h>

#define VELFILESUFFIX "vel"
#define DENFILESUFFIX "den"
#define TRIFILESUFFIX "tri"



class TriHeader{
public:
    uint64_t numOfTriangles;
    float boxSize;
    int other[61];
};

#endif