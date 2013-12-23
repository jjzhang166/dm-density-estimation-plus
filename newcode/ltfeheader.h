#ifndef __LY__LTFEHEADER__
#define __LY__LTFEHEADER__
struct LTFEHeader{
    int xyGridSize;
    int zGridSize;
    float boxSize;
    float startZ;
    float dz;
    char other[256 - 4 * 2 - 4 * 3];
};
#endif