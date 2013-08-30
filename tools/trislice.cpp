#include <fstream>
#include <string>
#include <sstream>
#include <iostream>
#include <cmath>

#include "tetrahedron.h"
#include "indtetrastream.h"
#include "tfileheader.h"
#include "triconverter.h"

using namespace std;

string prefix = "";
string base_name = "";
string singlefilename = "";
string outputPrefix = "";
string outputBaseName = "";
int parttype = 1;
int numoffiles = 0;
int datagridsize = -1;
int inputmemgrid = 16;
int imageSize = 1024;
double boxSize = 32000;


void printUsage(string pname){
    printf("Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n",
           pname.c_str(),
           "-df <single_gadgetfile name>",
           "-mf <prefix> <basename> <numoffiles>",
           "-of <outputprefix> <outputbasename>",
           "-imsize <imagesize>",
           "-parttype <particletype>, default: -1",
           "-dgridsize <data gridsize>, default: -1",
           "-tgrid <grid in memory for tetra>, default: -1"
           );
}

void savefile(TetraStreamer &streamer){
    TriConverter triangleConverter(imageSize,
                 streamer.getIndTetraStream()->getHeader().BoxSize,
                 outputPrefix,
                 outputBaseName);
    //printf("ok3\n");
    
    if(datagridsize == -1){
        datagridsize = (int)ceil(pow(streamer.getIndTetraStream()->getHeader().npartTotal[parttype], 1.0 / 3.0));
    }
    
    if(inputmemgrid == -1){
		inputmemgrid = datagridsize;
	}
   
     
    int numTetras = 0;

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
            {
                triangleConverter.process(tetras[i]);
                //tetrabuffer[numTetras] = tetras[i];
            }
            
                        
            if((tetra_count %  tcount )== 0){
                printf(">");
                cout.flush();
            }
            tetra_count ++;
        }

    }
    triangleConverter.finish();
    
    numTetras = 0;
    printf("\nFinished. In total %ld tetrahedrons output.\n", (long) tetra_count);
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
                outputPrefix = args[k+1];
                //printf("%s\n", outputPrefix);
                k++;
                outputBaseName = args[k+1];
            }else if(strcmp(args[k], "-dgridsize") == 0){
                ss << args[k + 1];
                ss >> datagridsize;
            }else if(strcmp(args[k], "-parttype") == 0){
                ss << args[k + 1];
                ss >> parttype;
            }else if(strcmp(args[k], "-tgrid") == 0){
                ss << args[k + 1];
                ss >> inputmemgrid;
            }else if(strcmp(args[k], "-imsize") == 0){
                ss << args[k + 1];
                ss >> imageSize;
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
        //printf("ok2\n");
        savefile(streamer);
        
    }
}
