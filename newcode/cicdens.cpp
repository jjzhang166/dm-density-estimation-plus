#include <fstream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cmath>


#include "GadgetReader/gadgetreader.hpp"
#include "GadgetReader/gadgetheader.h"
#include "ltfeheader.h"
#include "processbar.h"
#include "cic.h"

using namespace std;

#define BUFFERSIZE (1024*1024*16)

string base_name = "";

string outputdensfile = "";
string outputvelxfile = "";
string outputvelyfile = "";
string outputvelzfile = "";

bool isDens = false;
bool isVelx = false;
bool isVely = false;
bool isVelz = false;

bool isVdisp = false;

int imageSize = 128;
int gridSize = -1;
int partype = 1;
int64_t numParts = 0;
bool isVelocity = false;

void printUsage(string pname){
    printf("Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n"
           " %s\n %s\n %s\n",
           pname.c_str(),
           "-df <basename>",
           "-dens <outputdensfile>",
           "-velx <outputvelxfile>",
           "-vely <outputvelyfile>",
           "-velz <outputvelzfile>",
           "-partype <part type>",
           "-vdisp calculate the velocity dispersion in v-field",
           "-dgridsize <datagridsize (numparts^(1/3))>",
           "-imsize <imagesize>"
           );
}

int main(int argv, char * args[]){
    int numOfOutputs = 0;
    int k = 1;
    if(argv == 1){
        printUsage(args[0]);
        exit(1);
    }else{
        while(k < argv){
            stringstream ss;
            //printf("%s\n", args[k]);
            if(strcmp(args[k], "-df") == 0){
                base_name = args[k + 1];
            }else if(strcmp(args[k], "-dens") == 0){
                outputdensfile = args[k+1];
                isDens = true;
                numOfOutputs ++;
            }else if(strcmp(args[k], "-velx") == 0){
                outputvelxfile = args[k+1];
                isVelocity = true;
                isVelx = true;
                numOfOutputs ++;
            }else if(strcmp(args[k], "-vely") == 0){
                outputvelyfile = args[k+1];
                isVelocity = true;
                isVely = true;
                numOfOutputs ++;
            }else if(strcmp(args[k], "-velz") == 0){
                outputvelzfile = args[k+1];
                isVelocity = true;
                isVelz = true;
                numOfOutputs ++;
            }else if(strcmp(args[k], "-parttype") == 0){
                ss << args[k + 1];
                ss >> partype;
            }else if(strcmp(args[k], "-vdisp") == 0){
                isVdisp = true;
                k --;
            }else if(strcmp(args[k], "-dgridsize") == 0){
                ss << args[k + 1];
                ss >> gridSize;
            }else if(strcmp(args[k], "-imsize") == 0){
                ss << args[k + 1];
                ss >> imageSize;
            }else{
                printUsage(args[0]);
                exit(1);
            }
            k += 2;
        }
    }
    


    printf("Input File: %s\n", base_name.c_str());
    if(isDens){
        printf("Output Density: %s\n", outputdensfile.c_str());
    }
    
    if(isVelx){
        printf("Output Velocity X: %s\n", outputvelxfile.c_str());
    }
    
    if(isVely){
        printf("Output Velocity Y: %s\n", outputvelyfile.c_str());
    }
    
    if(isVelz){
        printf("Output Velocity Z: %s\n", outputvelzfile.c_str());
    }
    
    printf("Particle Type: %d\n", partype);
    
    
    
    unsigned int ignorecode = ~(1 << partype);
    GadgetReader::GSnap snap(base_name, false);
    
    int64_t nparts = snap.GetNpart(partype);
    
    if(gridSize == -1){
        gridSize = ceil(pow(nparts, 1.0 / 3.0));
    }
    
    fprintf(stderr, "Num of Particles: %lld\n", nparts);
    fprintf(stderr, "BoxSize: %f\n", snap.GetHeader().BoxSize);
    fprintf(stderr, "Num Files: %d\n", snap.GetNumFiles());
    fprintf(stderr, "Grid Size: %d\n", gridSize);
    fprintf(stderr, "Rendering ...\n");
    
    
    
    CIC cic(snap.GetHeader().BoxSize, imageSize, isVelocity, isVdisp);
    ProcessBar bar(nparts, 0);
    bar.start();
    
    
    int64_t cts = 0;
    while(cts < nparts){
        vector<float> temppos = snap.GetBlock("POS ", BUFFERSIZE, cts, ignorecode);
        vector<float> tempvel = snap.GetBlock("VEL ", BUFFERSIZE, cts, ignorecode);
        cts += temppos.size() / 3;
        bar.setvalue(cts);
        for(int i = 0; i < temppos.size() / 3; i++){
            double pos[3],vel[3];
            
            pos[0] = temppos[i * 3 + 0];
            pos[1] = temppos[i * 3 + 1];
            pos[2] = temppos[i * 3 + 2];
            
            vel[0] = tempvel[i * 3 + 0];
            vel[1] = tempvel[i * 3 + 1];
            vel[2] = tempvel[i * 3 + 2];
            
            cic.render_particle(pos, vel);
        }
    }
    
    

    bar.end();
    
    
    LTFEHeader lheader;
    lheader.xyGridSize = imageSize;
    lheader.zGridSize = imageSize;
    lheader.boxSize = snap.GetHeader().BoxSize;
    lheader.startZ = 0;
    lheader.dz = snap.GetHeader().BoxSize/ imageSize;
    
    printf("Writing to files.\n");
    
    if(isDens){
        fstream outputStream_;
        outputStream_.open(outputdensfile.c_str(), ios::out | ios::binary);
        if(!outputStream_.good()){
            fprintf(stderr, "Output Density File Error: %s !\n", outputdensfile.c_str());
            exit(1);
        }
        outputStream_.write((char *) &lheader, sizeof(LTFEHeader));
        
        double * densdata = cic.getDensityField();
        for(int i = 0; i < imageSize*imageSize*imageSize; i++){
            float dens = densdata[i];
            outputStream_.write((char *) &dens, sizeof(float));
        }
    }
    
    if(outputvelxfile != ""){
        fstream outputStream_;
        outputStream_.open(outputvelxfile.c_str(), ios::out | ios::binary);
        if(!outputStream_.good()){
            fprintf(stderr, "Output Velocity X File Error: %s !\n", outputvelxfile.c_str());
            exit(1);
        }
        outputStream_.write((char *) &lheader, sizeof(LTFEHeader));
        
        double * veldata = cic.getVelocityXField();
        for(int i = 0; i < imageSize*imageSize*imageSize; i++){
            float val = veldata[i];
            outputStream_.write((char *) &val, sizeof(float));
        }
    }
    
    if(outputvelyfile != ""){
        fstream outputStream_;
        outputStream_.open(outputvelyfile.c_str(), ios::out | ios::binary);
        if(!outputStream_.good()){
            fprintf(stderr, "Output Velocity Y File Error: %s !\n", outputvelyfile.c_str());
            exit(1);
        }
        outputStream_.write((char *) &lheader, sizeof(LTFEHeader));
        
        double * veldata = cic.getVelocityYField();
        for(int i = 0; i < imageSize*imageSize*imageSize; i++){
            float val = veldata[i];
            outputStream_.write((char *) &val, sizeof(float));
        }
    }

    
    if(outputvelzfile != ""){
        fstream outputStream_;
        outputStream_.open(outputvelzfile.c_str(), ios::out | ios::binary);
        if(!outputStream_.good()){
            fprintf(stderr, "Output Velocity Z File Error: %s !\n", outputvelzfile.c_str());
            exit(1);
        }
        outputStream_.write((char *) &lheader, sizeof(LTFEHeader));
        
        double * veldata = cic.getVelocityZField();
        for(int i = 0; i < imageSize*imageSize*imageSize; i++){
            float val = veldata[i];
            outputStream_.write((char *) &val, sizeof(float));
        }
    }

    
}
