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

using namespace std;

string prefix = "";
string base_name = "";
int numOfFiles = 0;
string outputfilename[4];

string outputdensfile = "";
string outputvelxfile = "";
string outputvelyfile = "";
string outputvelzfile = "";

int imageSize = 128;

int fileFloats[] = {1, 3, 3, 3};  //dens, velx, vely, velz
int floatOfVerts[4];
string compSuffix[4];
string componentFiles[4];

void printUsage(string pname){
    printf("Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n",
           pname.c_str(),
           "-df <basename>",
           "-dens <outputdensfile>",
           "-velx <outputvelxfile>",
           "-vely <outputvelyfile>",
           "-velz <outputvelzfile>",
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
                outputfilename[numOfOutputs] = outputdensfile;
                floatOfVerts[numOfOutputs] = fileFloats[0];
                compSuffix[numOfOutputs] = DENFILESUFFIX;
                numOfOutputs ++;
            }else if(strcmp(args[k], "-velx") == 0){
                outputvelxfile = args[k+1];
                outputfilename[numOfOutputs] = outputvelxfile;
                floatOfVerts[numOfOutputs] = fileFloats[1];
                compSuffix[numOfOutputs] = VELXFILESUFFIX;
                numOfOutputs ++;
            }else if(strcmp(args[k], "-vely") == 0){
                outputvelyfile = args[k+1];
                outputfilename[numOfOutputs] = outputvelyfile;
                floatOfVerts[numOfOutputs] = fileFloats[2];
                compSuffix[numOfOutputs] = VELYFILESUFFIX;
                numOfOutputs ++;
            }else if(strcmp(args[k], "-velz") == 0){
                outputvelzfile = args[k+1];
                outputfilename[numOfOutputs] = outputvelzfile;
                floatOfVerts[numOfOutputs] = fileFloats[3];
                compSuffix[numOfOutputs] = VELZFILESUFFIX;
                numOfOutputs ++;
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
    


    
    TrifileReader reader(base_name);
    //printf("OK\n");
    
    if(reader.getHeader().numOfZPlanes < imageSize){
        fprintf(stderr, "Num of zplanes in files less than imagesize.\n");
        exit(1);
    }
    if(reader.getHeader().numOfZPlanes % imageSize != 0){
        fprintf(stderr, "Image size is not a divisor of numOfFilesß.\n");
        exit(1);
    }
    
    
    TriDenRender render(imageSize,
                        reader.getHeader().boxSize,
                        outputfilename,
                        numOfOutputs
                        );
    
    

    if(!reader.isOpen()){
        printf("Input File Incorrect!\n");
        exit(1);
    }

    //printf("ImageSize: %d", imageSize);
    
    ProcessBar bar(imageSize, 0);
    bar.start();
    
    //int tcount = imageSize / 20;
    for(int i = 0; i < imageSize; i++){
        bar.setvalue(i);
        //int fileno = i * numOfFiles / imageSize;
        
        int plane = reader.getHeader().numOfZPlanes / imageSize * i;
        
        reader.loadPlane(plane);
        //printf("ok2.5\n");
        render.rend(reader.getTriangles(plane), reader.getDensity(plane), reader.getNumTriangles(plane));
    }

    bar.end();
    render.close();
    printf("Done.\n");
    
}
