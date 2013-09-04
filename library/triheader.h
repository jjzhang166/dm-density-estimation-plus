#ifndef __TRIHEADER__
#define __TRIHEADER__
#include <stdint.h>

#define VELXFILESUFFIX "velx"
#define VELYFILESUFFIX "vely"
#define VELZFILESUFFIX "velz"
#define DENFILESUFFIX "den"
#define TRIFILESUFFIX "tri"



class TriHeader{
public:
    uint64_t numOfTriangles;
    float boxSize;
    int other[61];
};

#endif