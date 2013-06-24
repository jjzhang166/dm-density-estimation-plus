#ifndef __HALOREAD__
#define __HALOREAD__

#include "../library/types.h"

int getHaloById(const char * fitsname, int halonum, Halo * halo);
int getHaloById(fitsfile * fptr, int halonum, Halo * halo);
int getTotalHaloNum(const char * fitsname);

#endif