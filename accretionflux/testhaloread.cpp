#include <cstring>
#include <cstdio>
#include <cstdlib>     /* atoi */
#include <fitsio.h>
#include "haloread.h"
#include "../library/types.h"

using namespace std;

int main(int argc, char *argv[])
{
    if (argc != 3) {
        printf("Usage:  testhaloread filename #halo\n");
        printf("\n");
        printf("The the halo information of the specific halo numer\n");
        printf("The input file must be a correct halo catalog file\n");
        return(0);
    }
    int haloid = atoi(argv[2]);
    Halo halo;
    
    int numhalo = getTotalHaloNum(argv[1]);
    printf("Number of Halos: %d\n", numhalo);
    
    int status = getHaloById(argv[1], haloid, &halo);
    if(status == 0){
        printf("x=%f\n", halo.x);
        printf("y=%f\n", halo.y);
        printf("z=%f\n", halo.z);
        printf("vx=%f\n", halo.vx);
        printf("vy=%f\n", halo.vy);
        printf("vz=%f\n", halo.vz);
        printf("lx=%f\n", halo.lx);
        printf("ly=%f\n", halo.ly);
        printf("lz=%f\n", halo.lz);
        printf("i1=%f\n", halo.i1);
        printf("i2=%f\n", halo.i2);
        printf("i3=%f\n", halo.i3);
        printf("i1x=%f\n", halo.i1x);
        printf("i1y=%f\n", halo.i1y);
        printf("i1z=%f\n", halo.i1z);
        printf("i2x=%f\n", halo.i2x);
        printf("i2y=%f\n", halo.i2y);
        printf("i2z=%f\n", halo.i2z);
        printf("i3x=%f\n", halo.i3x);
        printf("i3y=%f\n", halo.i3y);
        printf("i3z=%f\n", halo.i3z);
        printf("mass=%f\n", halo.mass);
        printf("radius=%f\n", halo.radius);
        printf("multi=%d\n", halo.multi);
        printf("v_max=%f\n", halo.v_max);
        printf("m_max=%f\n", halo.m_max);
        printf("r_max=%f\n", halo.r_max);
        printf("parent_id=%d\n", halo.parent_id);
        printf("parent_common=%f\n", halo.parent_common);
    }
    
    return status;
}

