/****************************************************************
 * This file contains the main function of rendering the triangle
 * slices of tetrahedra tessellation into density cube grid.
 *
 * Author: Lin F. Yang
 * Date: Feb. 2014
 ****************************************************************/

#include <fstream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cmath>

#include "tetrahedron.h"
#include "triheader.h"
#include "trirender.h"
#include "trifile_util.h"
#include "processbar.h"
#include "ltfeheader.h"

using namespace std;

string prefix = "";
string base_name = "";
int numOfFiles = 0;

string outputdensfile = "";
string outputvelxfile = "";
string outputvelyfile = "";
string outputvelzfile = "";

int imageSize = 256;
bool isVelocity = false;
bool isDensity =false;
bool isVelx = false;
bool isVely = false;
bool isVelz = false;
bool isVelDisp = false;

void printUsage(string pname){
    printf("Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n",
           pname.c_str(),
           "-df <basename>",
           "-dens <outputdensfile>",
           "-velx <outputvelxfile>",
           "-vely <outputvelyfile>",
           "-velz <outputvelzfile>",
           "-vdisp to measure the velocity^2 of velocity field",
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
            if(strcmp(args[k], "-df") == 0){
                base_name = args[k + 1];
            }else if(strcmp(args[k], "-dens") == 0){
                outputdensfile = args[k+1];
                isDensity = true;
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
            }else if(strcmp(args[k], "-imsize") == 0){
                ss << args[k + 1];
                ss >> imageSize;
            }else if(strcmp(args[k], "-vdisp") == 0){
                isVelDisp = true;
                k --;
            }else{
                printUsage(args[0]);
                exit(1);
            }
            k += 2;
        }
    }
    


    
    TrifileReader reader(base_name, isVelocity);
    
    if(reader.getHeader().numOfZPlanes < imageSize){
        fprintf(stderr, "Num of zplanes in files less than imagesize.\n");
        exit(1);
    }
    if(reader.getHeader().numOfZPlanes % imageSize != 0){
        fprintf(stderr, "Image size is not a divisor of numOfFilesß.\n");
        exit(1);
    }
   
    /* Initialize the render */
    TriDenRender render(imageSize,
                        reader.getHeader().boxSize
                        );
    

    if(!reader.isOpen()){
        printf("Input File Incorrect!\n");
        exit(1);
    }

    
    /* Output file streams */
    fstream outputDensStream_, outputVelXStream_,
        outputVelYStream_, outputVelZStream_;
    
    bool isL1File = false;
   

    /* Open the ouput streams */
    if(isDensity){
        outputDensStream_.open(outputdensfile.c_str(), ios::out | ios::binary);
        if(!outputDensStream_.good()){
            fprintf(stderr, "Output Density File Error: %s !\n", outputdensfile.c_str());
            exit(1);
        }
        isL1File = true;
    }
    
    if(isVelx){
        outputVelXStream_.open(outputvelxfile.c_str(), ios::out | ios::binary);
        if(!outputVelXStream_.good()){
            fprintf(stderr, "Output Velocity X File Error: %s !\n", outputvelxfile.c_str());
            exit(1);
        }
        isL1File = true;
    }
    
    if(isVely){
        outputVelYStream_.open(outputvelyfile.c_str(), ios::out | ios::binary);
        if(!outputVelYStream_.good()){
            fprintf(stderr, "Output Velocity Y File Error: %s !\n", outputvelyfile.c_str());
            exit(1);
        }
        isL1File = true;
    }
    
    if(isVelz){
        outputVelZStream_.open(outputvelzfile.c_str(), ios::out | ios::binary);
        if(!outputVelZStream_.good()){
            fprintf(stderr, "Output Velocity Z File Error: %s !\n", outputvelzfile.c_str());
            exit(1);
        }
        isL1File = true;
    }
    
    if(! isL1File){
        fprintf(stderr, "No output file!\n");
        exit(1);
    }
    
    
    /* LTFE output header */
    LTFEHeader lheader;
    lheader.xyGridSize = imageSize;
    lheader.zGridSize = imageSize;
    lheader.boxSize = reader.getHeader().boxSize;
    lheader.startZ = 0;
    lheader.dz = reader.getHeader().boxSize / imageSize;
    
    if(isDensity){
        outputDensStream_.write((char *) &lheader, sizeof(LTFEHeader));
    }
    if(isVelx){
        outputVelXStream_.write((char *) &lheader, sizeof(LTFEHeader));
    }
    if(isVely){
        outputVelYStream_.write((char *) &lheader, sizeof(LTFEHeader));
    }
    if(isVelz){
        outputVelZStream_.write((char *) &lheader, sizeof(LTFEHeader));
    }
    
    
    
    ProcessBar bar(imageSize, 0);
    bar.start();
    
    int numtris = 0;

    /* Render the triangles, a plane a time */
    for(int i = 0; i < imageSize; i++){
        bar.setvalue(i);
       
        /* Calculate the actual plane id in the triangle file */
        int plane = reader.getHeader().numOfZPlanes * i / imageSize;
       
        /* Load the triangles into memory */
        reader.loadPlane(plane);
        
        
        /* Render the density */
        if(!isVelocity){
            render.rendDensity(reader.getTriangles(),
                               reader.getDensity(),
                               reader.getNumTriangles(plane));
        }else{
            render.rendDensity(reader.getTriangles(),
                               reader.getDensity(),
                               reader.getVelocityX(),
                               reader.getVelocityY(),
                               reader.getVelocityZ(),
                               reader.getNumTriangles(plane),
                               isVelDisp);
        }
        
        numtris += reader.getNumTriangles(plane);
        
        
        if(isDensity){
            outputDensStream_.write((char *) render.getDensity(), sizeof(float) * imageSize * imageSize);
        }
        if(isVelx){
            outputVelXStream_.write((char *) render.getVelocityX(), sizeof(float) * imageSize * imageSize);
        }
        if(isVely){
            outputVelYStream_.write((char *) render.getVelocityY(), sizeof(float) * imageSize * imageSize);
        }
        if(isVelz){
            outputVelZStream_.write((char *) render.getVelocityZ(), sizeof(float) * imageSize * imageSize);
        }
        
    }
    
    
    if(isDensity){
        outputDensStream_.close();
    }
    if(isVelx){
        outputVelXStream_.close();
    }
    if(isVely){
        outputVelYStream_.close();
    }
    if(isVelz){
        outputVelZStream_.close();
    }
    //outputStream_.close();

    bar.end();
    //render.close();
    printf("Done. %d triangles rendered.\n", numtris);
    
}
