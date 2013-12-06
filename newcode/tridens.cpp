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
           "-df <prefix> <basename> <numfiles>",
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
                prefix = args[k + 1];
                k++;
                base_name = args[k + 1];
                k++;
                numOfFiles = atoi(args[k + 1]);
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
    
    if(numOfFiles < imageSize){
        fprintf(stderr, "Num of triangle files less than imagesize.\n");
        exit(1);
    }
    
    if(numOfFiles % imageSize != 0){
        fprintf(stderr, "Image size is not a divisor of numOfFilesß.\n");
        exit(1);
    }
    
    TriHeader header;
    string firstfile = prefix + base_name + "."TRIFILESUFFIX".0";
    fstream headgetter(firstfile.c_str(), ios::in | ios::binary);
    if(!headgetter.good()){
        printf("File: %s corrupted!\n", firstfile.c_str());
    }
    headgetter.read((char*)&header, sizeof(TriHeader));
    headgetter.close();


    TriDenRender render(imageSize,
                        header.boxSize,
                        outputfilename,
                        numOfOutputs
                        );
    

    
    int tcount = imageSize / 20;
    for(int i = 0; i < imageSize; i++){
        int fileno = i * numOfFiles / imageSize;
        
        stringstream ss;
        ss << fileno;
        string trifile = prefix + base_name + "."TRIFILESUFFIX"." + ss.str();
        
        for(int j = 0; j < numOfOutputs; j++){
            //string denfile = prefix + base_name + "."DENFILESUFFIX"." + ss.str();
            componentFiles[j] = prefix + base_name + "."
                                + compSuffix[j] + "."
                                + ss.str();
        }
        
            //printf("ok2\n");
        render.rend(trifile, componentFiles, floatOfVerts);
            //printf("ok3\n");
        
        if(fileno % tcount == 0){
            printf(">");
            cout.flush();
        }
    }
    printf("\n");
    render.close();
    
}
