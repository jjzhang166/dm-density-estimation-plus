#ifndef __DENSHEADER__
#define __DENSHEADER__
#include <stdint.h>

class LTFEHeader{
public:
    int32_t gridSizeX;      //grid size of x
    int32_t gridSizeZ;      //grid size of y
	double boxSize;
    double startZ;          //which is the z-coordinates of the first frame
    double dz;              //the z-coor difference between 2 frames
    double redshift;        //the redshift of this file
    int32_t gridSizeY;      //grid size of z
    char other[256 - 4 * 3 - 4 * 8];
};

#endif