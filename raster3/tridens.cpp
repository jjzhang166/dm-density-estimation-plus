#include <fstream>
#include <cstdlib>
#include <string>
#include <cstring>
#include <sstream>
#include <iostream>
#include <cmath>

#include "tetrahedron.h"
#include "indtetrastream.h"
#include "triheader.h"
#include "trirender.h"

using namespace std;

string prefix = "";
string base_name = "";
int numOfFiles = 0;
string outputfilename = "";
int imageSize = 128;


void printUsage(string pname){
    printf("Usage: %s\n %s\n %s\n %s\n",
           pname.c_str(),
           "-df <prefix> <basename> <numfiles>",
           "-dens <outputfilename>",
           "-imsize <imagesize>"
           );
}

int main(int argv, char * args[]){
    
    
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
                outputfilename = args[k+1];
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
    
    TriDenRender render(imageSize, outputfilename, header.boxSize);
    
    int tcount = imageSize / 20;
    for(int i = 0; i < imageSize; i++){
        int fileno = i * numOfFiles / imageSize;
        
        stringstream ss;
        ss << fileno;
        string trifile = prefix + base_name + "."TRIFILESUFFIX"." + ss.str();
        string denfile = prefix + base_name + "."DENFILESUFFIX"." + ss.str();
        
        render.rend(trifile, denfile);
        
        if(fileno % tcount == 0){
            printf(">");
            cout.flush();
        }
    }
    printf("\n");
    render.close();
}
