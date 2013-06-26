/**
* Produce a mask file for the halos from the particle mask
* Input is a particle ;; and the gadget file
* Input maskfile is a binary file with 256 byte header, the first
* integer specifies the grid size
* Output is a binary halomask file, with 256 byte header and data
* The first integer of the 256 byte is total number of halos
*/


#include <cstring>
#include <cstdio>
#include <cstdlib>     /* atoi */
#include <sstream>
#include <fstream>
#include <vector>
#include <iostream>
#include <fitsio.h>

#include "types.h"
#include "tetrahedron.h"
#include "indtetrastream.h"

#include "accretion.h"
#include "haloread.h"
#include "../library/types.h"

using namespace std;

int main(int argc, char * argv[]){
    string partmaskfile;
    string halofile;
    string gadfile;
    string outputfile;
    
    char * partmask;
    GSnap * gsnap_;
    char * halomask;
    
    
    //if use -1, then use the particle gridsize as the gridsize
    //otherwise use the user setting
    int datagridsize = 256;
    //the particle type in the gadget file
    int parttype = 1;
    int inputmemgrid = 32;
    //load all the particles into memory?
    bool isHighMem = true;
    //return the all particle pointer?
    bool isAllData = false;

    
    
    if(argc != 5){
        printf("Usage: maskprod gadfile particlemask halocatlog outputmask\n");
        printf("Convert the particle mask file to be the halo mask file\n");
        exit(1);
    }
    
    gadfile = argv[1];
    partmaskfile = argv[2];
    halofile = argv[3];
    outputfile = argv[4];
    
    int head[64];
    int pargridsize;
    
    fstream partfst(partmaskfile.c_str(), ios::in | ios::binary);
    if(partfst.good()){
        partfst.read((char *)head, sizeof(int) * 64); 
    }else{
        printf("Particle Mask file incorrect!\n");
        exit(1);
    }
    pargridsize = head[0];
    
    gsnap_ = new GSnap(gadfile, isHighMem, parttype, datagridsize);
    int numparts = gsnap_->Npart;
    
    if(numparts != pargridsize * pargridsize * pargridsize){
        printf("Mask file does not fit with the gadfile!\n");
        exit(1);
    }
    
    partmask = new char[numparts]();
    if(partfst.good()){
        partfst.read((char *)partmask, numparts);
    }else{
        printf("Particle Mask file incorrect!\n");
        exit(1);
    }
    partfst.close();
    
    double mass = gsnap_->header.mass[1];
    
    Point * pos = gsnap_ -> getAllPos();//new Point[numparts];
    Point * vel = gsnap_ -> getAllVel();//new Point[numparts];
    
    //total halo number
    int hmax = getTotalHaloNum(halofile.c_str());
    
    
    halomask = new char[hmax]();
    
    printf("Start calculating...\n");

    for(int j = 0; j < numparts; j++){
        if(partmask[j] != 0){
            for(int i = 0; i < hmax; i++){
                if(halomask[i] == 0){
                    Halo halo;
                    int haloid = i + 1;
                    int status = getHaloById(halofile.c_str(), haloid, &halo);
                    if(status != 0){
                        printf("Unkown error!\n");
                        exit(1);
                    }
                    
                    Point hc;
                    hc.x = halo.x;
                    hc.y = halo.y;
                    hc.z = halo.z;
                    
                    Point rvec = pos[j] - hc;
                    double r = sqrt(rvec.dot(rvec));
                    
                    if(r < halo.radius){
                        halomask[i] = 1;
                    }
                }
            }
        }
    }
    

    fstream outfst(outputfile.c_str(), ios::out | ios::binary);
    head[0] = hmax;
    if(outfst.good()){
        outfst.write((char *) head, sizeof(int) * 64);
        outfst.write((char *) halomask, hmax);
    }
    
    outfst.close();
    
    printf("Finished...\n");
    delete partmask;
    delete gsnap_;
    delete halomask;
    
}
