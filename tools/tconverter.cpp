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
    printf("Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n",
           pname.c_str(),
           "-df <single_gadgetfile name>",
           "-mf <prefix> <basename> <numoffiles>",
           "-of <output t-file name>",
           "-parttype <particletype>, default: -1",
           "-dgridsize <data gridsize>, default: -1",
           "-tgrid <grid in memory for tetra>, default: -1" 
           );
}


void savefile(TetraStreamer &streamer){
    if(datagridsize == -1){
        datagridsize = (int)ceil(pow(streamer.getIndTetraStream()->getHeader().npartTotal[parttype], 1.0 / 3.0));
    }
    
    if(inputmemgrid == -1){
		inputmemgrid = datagridsize;
	}
   
     
    
    TFileHeader header;
    header.numOfTetrahedrons = 0l;
    header.boxSize = streamer.getIndTetraStream()->getHeader().BoxSize;
    int numTetras = 0;
   
    
    fstream outputstream(outputfile.c_str(), ios::out | ios::binary);
    if(!outputstream.good()){
        printf("Output file incorrect!!\n");
        exit(1);
    }
    
    uint64_t tetra_count = 0;
    uint64_t tcount = datagridsize * datagridsize * datagridsize / 10 * 6;
    if(tcount == 0){
        tcount = 1;
    }

    //printf("%d %d \n", tcount, datagridsize);
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
    printf("\nFinished. In total %ld tetrahedrons output.\n", tetra_count);
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
                ss << args[k + 1];
                ss >> singlefilename;
            }else if(strcmp(args[k], "-mf") == 0){
                prefix = args[k + 1];
                k++;
                base_name = args[k + 1];
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
            }else if(strcmp(args[k], "-tgrid") == 0){
                ss << args[k + 1];
                ss >> inputmemgrid;
            }else{
                printUsage(args[0]);
                exit(1);
            }
            k += 2;
        }
        //printf("%d\n", numoffiles);

        if(singlefilename == "" && numoffiles == 0){
            printUsage(args[0]);
            exit(1);
        }
    }
    
    
    

    
    if(numoffiles != 0){
       //printf("ok1\n"); 
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
       //printf("ok2\n"); 
       savefile(streamer);
        
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
