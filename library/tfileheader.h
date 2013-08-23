#ifndef __LYTFILEHEADER__
#define __LYTFILEHEADER__
#include <stdint.h>



//a file contains only tetrahedrons


class TFileHeader{
public:
    
//#if defined(_WIN32) || defined(_WIN64)
//	unsigned __int64 numOfTetrahedrons;
//#else
	uint64_t numOfTetrahedrons;
//#endif
    //
	float boxSize;
    int other[60];
};

#endif

