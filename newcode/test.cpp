#include "GadgetReader/gadgetreader.hpp"
#include "GadgetReader/gadgetheader.h"
#include <string>
#include <cstdlib>
#include <cmath>
#include <stdint.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <sstream>

#include "io_utils.h"


using namespace std;
using namespace GadgetReader;

int main(){
    //printf("OK\n");
    
    GSnap snap("/Users/lyang/data/multires_150", false);
    int64_t nparts = snap.GetNpart(1);
    printf("Nparts %d\n", nparts);
    
    vector<float> temppos = snap.GetBlock("POS ", nparts, 0, 0);
    float * pos = new float[temppos.size()];
    
    vector<float> tempvel = snap.GetBlock("VEL ", nparts, 0, 0);
    float * vel = new float[tempvel.size()];
    
    vector<long long> tempind = snap.GetBlockInt("ID  ", nparts, 0, 0);
    int64_t * inds = new int64_t[tempvel.size()];
    
    for(int i = 0; i < tempind.size(); i++){
        pos[3*tempind[i] + 0] = temppos[3*i + 0];
        pos[3*tempind[i] + 1] = temppos[3*i + 1];
        pos[3*tempind[i] + 2] = temppos[3*i + 2];
        
        vel[3*tempind[i] + 0] = tempvel[3*i + 0];
        vel[3*tempind[i] + 1] = tempvel[3*i + 1];
        vel[3*tempind[i] + 2] = tempvel[3*i + 2];
    }

    int numfiles = getNumDividerFiles("test/test");
    int gridsize = readDividerFileHeader("test/test", 0).gridsize;

    //testing
    for(int i = 0; i < numfiles; i++){
        int64_t numparts = getDividerFileNumParts("test/test", i);
        float * apos = new float[numparts * 3];
        float * avel = new float[numparts * 3];
        int64_t * ainds = new int64_t[numparts];
        

        readDividerFileInds("test/test", i, ainds);
        readDividerFilePos("test/test", i, apos);
        readDividerFileVel("test/test", i, avel);
        
        for(int j = 0; j < numparts; j++){
            int cid = ainds[j];
            if(cid >= gridsize * gridsize * gridsize)
                cid = cid - gridsize * gridsize * gridsize;
            
            bool vfal =
                (apos[3*j + 0] == pos[3*cid + 0]) &&
                (apos[3*j + 1] == pos[3*cid + 1]) &&
                (apos[3*j + 2] == pos[3*cid + 2]) &&
                (avel[3*j + 0] == vel[3*cid + 0]) &&
                (avel[3*j + 1] == vel[3*cid + 1]) &&
                (avel[3*j + 2] == vel[3*cid + 2]);
            printf("Index: %d. Num Parts: %d/%d\n", cid, numparts, gridsize * gridsize * gridsize);
            printf("%f %f %f %f %f %f\n"
                   "%f %f %f %f %f %f\n",
                   apos[3*j + 0],
                   apos[3*j + 1],
                   apos[3*j + 2],
                   avel[3*j + 0],
                   avel[3*j + 1],
                   avel[3*j + 2],
                   pos[3*cid + 0],
                   pos[3*cid + 1],
                   pos[3*cid + 2],
                   vel[3*cid + 0],
                   vel[3*cid + 1],
                   vel[3*cid + 2]);
            
            if(!vfal){
                printf("ERROR!!!\n");
                exit(1);
            }
        }
        
        delete[] apos;
        delete[] avel;
        delete[] ainds;
    }
    printf("NO ERROR!\n");
    delete[] pos;
    delete[] vel;
    delete[] inds;
    
}