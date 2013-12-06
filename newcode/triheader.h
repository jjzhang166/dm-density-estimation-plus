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
    int32_t fileID;
    int32_t NumFiles;
    int64_t NumTriangles;
    int32_t z_id;
	float boxSize;
    double z_coor;
    int other[256 - 4 - 4 - 8 - 4 - 8];
};

#endif