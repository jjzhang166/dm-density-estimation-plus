#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

#include "tetrahedron.h"
#include "indtetrastream.h"
#include "tfileheader.h"

#define BUFFERSIZE 65536

using namespace std;

string prefix = "";
string base_name = "";
string singlefilename = "";
string outputfile = "";
int parttype = 1;
int numoffiles = 0;
int datagridsize = -1;
int inputmemgrid = -1;
Tetrahedron tetrabuffer[BUFFERSIZE];

void printUsage(string pname){
    printf("Usage: %s\n %s\n %s\n %s\n %s\n %s\n",
           pname.c_str(),
           "-df <single_gadgetfile name>",
           "-mf <prefix> <basename> <numoffiles>",
           "-of <output t-file name>",
           "-parttype <particletype>, default: -1",
           "-dgridsize <data gridsize>, defualt: -1"
           );
}


void savefile(TetraStreamer &streamer){
    if(datagridsize == -1){
        datagridsize = ceil(pow(streamer.getIndTetraStream()->getHeader().npart[parttype], 1.0 / 3.0));
    }
    
    if(inputmemgrid == -1){
		inputmemgrid = datagridsize;
	}
    
    
    TFileHeader header;
    header.numOfTetrahedrons = 0;
    header.boxSize = streamer.getIndTetraStream()->getHeader().BoxSize;
    int numTetras = 0;
    
    fstream outputstream(outputfile.c_str(), ios::out | ios::binary);
    if(!outputstream.good()){
        printf("Output file incorrect!!\n");
        exit(1);
    }
    
    int tetra_count = 0;
    int tcount = datagridsize * datagridsize * datagridsize * 6 / 10;
    if(tcount == 0){
        tcount = 1;
    }
    streamer.reset();
    while(streamer.hasNext()){
        int nums;
        
        Tetrahedron * tetras;
        tetras = streamer.getNext(nums);

        for(int i= 0; i < nums; i++){
            tetrabuffer[numTetras] = tetras[i];
            numTetras ++;
            //printf("ok %d\n", tetra_count);
            if(numTetras >= BUFFERSIZE){
                //write to file
                outputstream.write((char *) tetrabuffer,
                                   numTetras * sizeof(Tetrahedron));
                numTetras = 0;
            }
            
            if((tetra_count %  tcount )== 0){
                printf(">");
                cout.flush();
            }
            tetra_count ++;
        }
        
        //printf("ok1 %d\n", tetra_count);
    }
    outputstream.write((char *) tetrabuffer,
                       numTetras * sizeof(Tetrahedron));
    numTetras = 0;
    header.numOfTetrahedrons = tetra_count;
    outputstream.seekg(0, outputstream.beg);
    outputstream.write((char *) &header, sizeof(TFileHeader));
    outputstream.close();
    printf("\nFinished. In total %d tetrahedrons output.\n", tetra_count);
}


int main(int argv, char * args[]){

    
    int k = 1;
    if(argv == 1){
        printUsage(args[0]);
        exit(1);
    }else{
        while(k < argv){
            stringstream ss;
            if(strcmp(args[k], "-df") == 0){
                ss << args[k + 1];
                ss >> singlefilename;
            }else if(strcmp(args[k], "-mf") == 0){
                ss << args[k + 1];
                ss >> prefix;
                k++;
                ss << args[k + 1];
                ss >> base_name;
                k++;
                ss << args[k + 1];
                ss >> numoffiles;
            }else if(strcmp(args[k], "-of") == 0){
                ss << args[k + 1];
                ss >> outputfile;
            }else if(strcmp(args[k], "-dgridsize") == 0){
                ss << args[k + 1];
                ss >> datagridsize;
            }else if(strcmp(args[k], "-parttype") == 0){
                ss << args[k + 1];
                ss >> parttype;
            }else{
                printUsage(args[0]);
                exit(1);
            }
            k += 2;
        }
        
        if(singlefilename == "" && numoffiles == 0){
            printUsage(args[0]);
            exit(1);
        }
    }
    
    
    

    
    if(numoffiles != 0){
        TetraStreamer streamer(prefix,
                               base_name,
                               numoffiles,
                               inputmemgrid,
                               parttype,
                               datagridsize,
                               true,
                               true,
                               true,
                               true,
                               false);
        
    }else{
        TetraStreamer streamer(singlefilename,
                               inputmemgrid,
                               parttype,
                               datagridsize,
                               true,
                               true,
                               true,
                               true,
                               false);
        savefile(streamer);
        
    }
}
