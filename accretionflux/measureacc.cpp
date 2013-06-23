#include <cstring>
#include <cstdio>
#include <cstdlib>     /* atoi */
#include <fitsio.h>
#include "haloread.h"
#include "../library/types.h"

using namespace std;

void printusage(){
    printf("Usage: measureacc datafile halocatalog [\\LTFE] [\\SHELL]\n");
    printf("\n");
    printf("Measure the accretion rate of halos in the datafile (gadget file)\n descripted by the halocatalog (fits file)\n");
    printf("\\LTFE: use LTFE method to do the measurement (default)");
    printf("\\SHELL: use shell method to do the measurement");
}

int main(int argc, char *argv[])
{
    string filename;
    string 
    if (argc < 3) {
        printusage();
        return(0);
    }else{
        
    }

}
