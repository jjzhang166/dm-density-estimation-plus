#include "GadgetReader/gadgetreader.hpp"
#include "GadgetReader/gadgetheader.h"
#include <string>
#include <cstdlib>
#include <cmath>
#include <stdint.h>

using namespace std;
using namespace GadgetReader;
int main(int argv, char * args[]){
    if(argv != 6 && argv != 7){
        printf("%s Filename NumSlice PartType OutputDir OutputBase [GridSize]\n", args[0]);
        exit(1);
    }
    string inputfilename = args[1];
    int numslice = atoi(args[2]);
    int parttype = atoi(args[3]);
    string outputdir = args[4];
    string outputbase = args[5];
    printf("Input File: %s\n", inputfilename.c_str());
    printf("Numbers of slices %d\n", numslice);
    printf("OutputDir: %s\n", outputdir.c_str());
    printf("OutputBase: %s\n", outputbase.c_str());
    printf("Particle Type: %d\n", parttype);
    int gridsize = -1;
    if(argv == 6){
        gridsize = atoi(args[6]);
    }
    
    //dividing

    GSnap snap(inputfilename, false);
    int64_t nparts = snap.GetNpart(parttype);
    if(gridsize == -1){
        gridsize = pow(nparts, 1.0/3.0);
    }
    
    printf("Number of Particles: %lld\n", nparts);
    printf("Grid Size: %d\n", gridsize);
    printf("Dividing ...\n");
}