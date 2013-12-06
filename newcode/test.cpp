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
#include "dtetrastream.h"


using namespace std;
using namespace GadgetReader;

int main(){
    //printf("OK\n");
    
    GSnap snap("/Users/lyang/data/multires_150", false);
    int64_t nparts = snap.GetNpart(1);
    printf("Nparts %lld\n", nparts);
    
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
            //printf("Index: %d. Num Parts: %d/%d\n", cid, numparts, gridsize * gridsize * gridsize);
            //printf("%f %f %f %f %f %f\n"
            //       "%f %f %f %f %f %f\n",
            //       apos[3*j + 0],
            //       apos[3*j + 1],
            //       apos[3*j + 2],
            //       avel[3*j + 0],
            //       avel[3*j + 1],
            //       avel[3*j + 2],
            //       pos[3*cid + 0],
            //       pos[3*cid + 1],
            //      pos[3*cid + 2],
           //        vel[3*cid + 0],
           //        vel[3*cid + 1],
           //        vel[3*cid + 2]);
            
            if(!vfal){
                printf("ERROR!!!\n");
                exit(1);
            }
        }
        
        delete[] apos;
        delete[] avel;
        delete[] ainds;
    }
    printf("NO ERROR In Reading...\n");
    delete[] pos;
    delete[] vel;
    delete[] inds;
    
    
    DtetraStream dstream("test/test");



    IndTetrahedron indtetra;
    int tetracount = 0;
    int tetraall_count = 0;
    
    for(int l = 0; l < numfiles; l++){
        dstream.loadBlock(l);
        int numindtetra = dstream.getNumTetras();
        printf("%d\n", numindtetra);
        IndTetrahedronManager & im = dstream.getCurrentIndtetraManeger();
        divide_header header = dstream.getHeader();
        
        for(int i = 0; i < numindtetra; i ++){
            dstream.getIndTetra(indtetra, i);
            tetracount ++;
            int i2 = indtetra.ind2  % gridsize - indtetra.ind1  % gridsize;
            int j2 = indtetra.ind2 / gridsize % gridsize - indtetra.ind1 / gridsize % gridsize;
            int k2 = indtetra.ind2 / gridsize / gridsize % gridsize - indtetra.ind1 / gridsize / gridsize % gridsize;
            
            int i3 = indtetra.ind3  % gridsize - indtetra.ind1  % gridsize;
            int j3 = indtetra.ind3 / gridsize % gridsize - indtetra.ind1 / gridsize % gridsize;
            int k3 = indtetra.ind3 / gridsize / gridsize % gridsize - indtetra.ind1 / gridsize / gridsize % gridsize;
            
            int i4 = indtetra.ind4  % gridsize - indtetra.ind1  % gridsize;
            int j4 = indtetra.ind4 / gridsize % gridsize - indtetra.ind1 / gridsize % gridsize;
            int k4 = indtetra.ind4 / gridsize / gridsize % gridsize - indtetra.ind1 / gridsize / gridsize % gridsize;
            
            if(i2 < 0) i2 += gridsize;
            if(j2 < 0) j2 += gridsize;
            if(k2 < 0) k2 += gridsize;
            if(i3 < 0) i3 += gridsize;
            if(j3 < 0) j3 += gridsize;
            if(k3 < 0) k3 += gridsize;
            if(i4 < 0) i4 += gridsize;
            if(j4 < 0) j4 += gridsize;
            if(k4 < 0) k4 += gridsize;
            
            //printf("%d %d %d %d\n", indtetra.ind1+header.startind,
            //       indtetra.ind2+header.startind, indtetra.ind3+header.startind,
            //       indtetra.ind4+header.startind );
            
            //printf("%d%d%d-%d%d%d-%d%d%d-%d%d%d\n",
            //       0,  0,  0,
            //       i2, j2, k2,
            //       i3, j3, k3,
            //       i4, j4, k4
            //       );
            
            
            
            printf("%f %f %f \n%f %f %f \n%f %f %f \n%f %f %f\n\n\n",
                          im.posa(indtetra).x, im.posa(indtetra).y, im.posa(indtetra).z,
                          im.posb(indtetra).x, im.posb(indtetra).y, im.posb(indtetra).z,
                          im.posc(indtetra).x, im.posc(indtetra).y, im.posc(indtetra).z,
                          im.posd(indtetra).x, im.posd(indtetra).y, im.posd(indtetra).z
                          );
            
            int nt = im.getNumPeriodical(indtetra);
            Tetrahedron * ts = im.getPeroidTetras(indtetra);
            for(int k = 0; k < nt; k++){
                tetraall_count ++;
                //printf("%f %f %f \n%f %f %f \n%f %f %f \n%f %f %f\n\n\n",
                //       ts[k].v1.x, ts[k].v1.y, ts[k].v1.z,
                //       ts[k].v2.x, ts[k].v2.y, ts[k].v2.z,
                //       ts[k].v3.x, ts[k].v3.y, ts[k].v3.z,
                //       ts[k].v4.x, ts[k].v4.y, ts[k].v4.z
                //       );
            }
        }
    }
    
    printf("Ind Tetras: %d =?= %d\n", tetracount, 6*gridsize*gridsize*gridsize);
    printf("Tetras: %d\n", tetraall_count);
    //for(int i = 0; i < numfiles; i ++){
    //
    //}
}