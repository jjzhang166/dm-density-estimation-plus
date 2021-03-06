//#define __STDC_FORMAT_MACROS
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <algorithm>    // std::find
#include <vector>
#include <stdint.h>


#include "denrender.h"
#include "tfilestream.h"
#include "tetrahedron.h"
#include "indtetrastream.h"


using namespace std;
string filename = "";
int blocksize = 819200;
int imagesize = 512;

string densityFilename = "";	//output filename
string velocityXFilename = "";	//velocity x output filename
string velocityYFilename = "";  //velocity y output filename
string velocityZFilename = "";  //velocity z output filename
string streamFilename = "";     //stream statistics filename

int mem_cut_limit = -1;         //for limit CPU memory, limit the number of cuts in memory

int numOfCuts = 0;
//to render a larger scene, rend several times for them

vector<RenderType> renderTypes; //what data component will be rendered
fstream * pOutputStreams; //what data component will be rendered

float startz = 0;
float dz = 0;


void printUsage(string pname){  //print the usage
    fprintf(stdout,
            "Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n"
            , pname.c_str()
            , "[-imsize <imagesize>] default: 512"
            , "[-df <tetra-stream file>]"
            , "[-dens <output density file>]"
            , "[-stream <output stream data file>]"
            , "[-velx <output velocity x-component file>]"
            , "[-vely <output velocity y-component file>]"
            , "[-velz <output velocity z-component file>]"
            , "[-memlimit <tetras stored in mem>] default: 65536"
            , "[-startz] the starting z-coordinates to calculate the cuts"
            , "[-dz] the interval between each 2 z-plane"
            , "[-numz] the number of z-planes"
            , "[-cutslimit] <limit the number of cuts in CPU memory> "
                "default: -1, no limit"
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
            if(strcmp(args[k], "-imsize") == 0){
                ss << args[k + 1];
                ss >> imagesize;
            }else if(strcmp(args[k], "-df") == 0){
                ss << args[k + 1];
                ss >> filename;
            }else if(strcmp(args[k], "-dens") == 0){
                ss << args[k + 1];
                ss >> densityFilename;
                if(densityFilename != "" &&
                   find (renderTypes.begin(), renderTypes.end(), DENSITY) == renderTypes.end()
                   ){
                    renderTypes.push_back(DENSITY);
                }
            }else if(strcmp(args[k], "-velx") == 0){
                ss << args[k + 1];
                ss >> velocityXFilename;
                if(velocityXFilename != "" &&
                   find (renderTypes.begin(), renderTypes.end(), VELOCITY_X) == renderTypes.end()
                   ){
                    renderTypes.push_back(VELOCITY_X);
                }
            }else if(strcmp(args[k], "-vely") == 0){
                ss << args[k + 1];
                ss >> velocityYFilename;
                if(velocityYFilename != ""  &&
                   find (renderTypes.begin(), renderTypes.end(), VELOCITY_Y) == renderTypes.end()
                   ){
                    renderTypes.push_back(VELOCITY_Y);
                }
            }else if(strcmp(args[k], "-velz") == 0){
                ss << args[k + 1];
                ss >> velocityZFilename;
                if(velocityZFilename != ""  &&
                   find (renderTypes.begin(), renderTypes.end(), VELOCITY_Z) == renderTypes.end()
                   ){
                    renderTypes.push_back(VELOCITY_Z);
                }
            }else if(strcmp(args[k], "-stream") == 0){
                ss << args[k + 1];
                ss >> streamFilename;
                if(streamFilename != ""  &&
                   find (renderTypes.begin(), renderTypes.end(), STREAM) == renderTypes.end()
                   ){
                    renderTypes.push_back(STREAM);
                }
            }else if(strcmp(args[k], "-startz") == 0){
                ss << args[k + 1];
                ss >> startz;
            }else if(strcmp(args[k], "-dz") == 0){
                ss << args[k + 1];
                ss >> dz;
            }else if(strcmp(args[k], "-numz") == 0){
                ss << args[k + 1];
                ss >> numOfCuts;
            }else if(strcmp(args[k], "-cutslimit") == 0){
                ss << args[k + 1];
                ss >> mem_cut_limit;
            }else if(strcmp(args[k], "-memlimit") == 0){
                ss << args[k + 1];
                ss >> blocksize;
            }else{
                printUsage(args[0]);
                exit(1);
            }
            k += 2;
        }
    }
    
    int numofrendertyps = renderTypes.size();
    if(numofrendertyps > DenRender::NUM_OF_RENDERTRYPE_LIMIT)
        numofrendertyps = DenRender::NUM_OF_RENDERTRYPE_LIMIT;
    
    
    string outputFilenames[] = {
        densityFilename,
        streamFilename,
        velocityXFilename,
        velocityYFilename,
        velocityZFilename
    };
    
    pOutputStreams = new fstream[numofrendertyps];

    TFileStream tfilestream(filename, blocksize);

    float boxsize = tfilestream.getHeader().boxSize;
    if(numOfCuts == 0){
        numOfCuts = imagesize;
        dz = boxsize / numOfCuts;
    }
    
    if(mem_cut_limit == -1){
        mem_cut_limit = numOfCuts;
    }
    
    uint64_t tetra_count = 0;
    int repeatTimes = (int)ceil((float) numOfCuts / (float) mem_cut_limit);
    int repeatNumOfCuts = numOfCuts > mem_cut_limit ?
        mem_cut_limit : numOfCuts;
    
    
    
    printf("\n=========================DENSITY ESTIMATION==========================\n");
    printf("*****************************PARAMETERES*****************************\n");
    printf("Render Image Size       = %d\n", imagesize);
	printf("Data File               = %s\n", filename.c_str());
    if(densityFilename != "")
        printf("Output File             = %s\n", densityFilename.c_str());
    
    if(streamFilename != "")
        printf("Stream Data File        = %s\n", streamFilename.c_str());
    
    if(velocityXFilename != "")
        printf("Velocity X File         = %s\n", velocityXFilename.c_str());
    
    if(velocityYFilename != "")
        printf("Velocity Y File         = %s\n", velocityYFilename.c_str());
    
    if(velocityZFilename != "")
        printf("Velocity Z File         = %s\n", velocityZFilename.c_str());
    printf("Rendering %d z-cuts of the density field. \n"
           "Start from z = %f, with dz = %f\n",
           numOfCuts, startz, dz);
    printf("*********************************************************************\n");
    
    
    
    
    //head used 256 bytes
    //the first is imagesize
    //the second the numOfCuts
    //the third is a float number boxsize
    //the 4-th is a float number startz
    //the 5-th is a fload number dz
    //All others are 0
    int head[59];
    
    for(int i = 0; i < numofrendertyps; i ++ ){
        if(outputFilenames[renderTypes[i]] != ""){
            pOutputStreams[i].open(outputFilenames[renderTypes[i]].c_str(),
                                    ios::out | ios::binary);
            while(!pOutputStreams[i].good()){
                printf("File error, calculation not saved for rendering type %d...!\n", renderTypes[i]);
                printf("Input new filename:\n");
                cin >> outputFilenames[renderTypes[i]];
                pOutputStreams[i].clear();
                pOutputStreams[i].open(outputFilenames[renderTypes[i]].c_str(), ios::out | ios::binary);
            }
            pOutputStreams[i].write((char *) &imagesize, sizeof(int));
            pOutputStreams[i].write((char *) &numOfCuts, sizeof(int));
            pOutputStreams[i].write((char *) &boxsize, sizeof(float));
            pOutputStreams[i].write((char *) &startz, sizeof(float));
            pOutputStreams[i].write((char *) &dz, sizeof(float));
            pOutputStreams[i].write((char *) head, sizeof(int) * 59);
        }
        
    }
    //delete outstreams;
    
    
    
    for(int _idcut = 0; _idcut < repeatTimes; _idcut ++){
        
        int newNumOfCuts = repeatNumOfCuts;
        if(newNumOfCuts * (_idcut + 1) > numOfCuts){
            newNumOfCuts = numOfCuts - mem_cut_limit * _idcut;
        }
        float newStartz = startz + _idcut * repeatNumOfCuts * dz;
        
        //printf("%f %f %d\n", newStartz, dz, newNumOfCuts);
        
        DenRender render(imagesize,
                         boxsize,
                         newStartz,
                         dz,
                         newNumOfCuts,
                         renderTypes);
        
        
        //render

		tfilestream.reset();
        
        uint64_t tcount = tfilestream.getNumofTetras() / 50;
        
        if(mem_cut_limit == numOfCuts){
			printf("Start rendering, %lld tetrahedrons ...\n", tfilestream.getNumofTetras());
        }else{
            printf("Rendering %d/%d, %lld tetrahedrons ...\n", _idcut + 1, repeatTimes, tfilestream.getNumofTetras());
        }
        
        
        while(tfilestream.hasNext()){
            int nums;
            Tetrahedron * tetras;
            tetras = tfilestream.getNext(nums);
            for(int i= 0; i < nums; i++){
				//test
                render.rend(tetras[i]);
                if((tetra_count %  tcount )== 0){
                    printf(">");
                    cout.flush();
                }
                tetra_count ++;
            }
        }
        render.finish();
        float * result = render.getResult();
        
        printf("\n");
        if(mem_cut_limit == numOfCuts){
            printf("Finished. In total %lld tetrahedron rendered.\n", tetra_count);
        }
        

        

        

        
        //fstream * outstreams = new fstream[numofrendertyps];
        
        printf("Saving ...\n");
        for(int i = 0; i < numofrendertyps; i ++ ){
            for(int j = 0; j < imagesize * imagesize * newNumOfCuts; j ++ ){
                if(pOutputStreams[i].good()){
					pOutputStreams[i].write((char *) (result + j * numofrendertyps + i),
                                        sizeof(float));
					//printf("%e\n", *(result + j * numofrendertyps + i));
				}else{
					printf("Output Error. Type %d!\n", i);
				}
            }
        }
        //delete outstreams;
        
    }
    
    for(int i = 0; i < numofrendertyps; i ++ ){
        pOutputStreams[i].close();
    }
    
    //outstream.open(gridfilename.c_str(), ios::out | ios::binary);
    if(mem_cut_limit != numOfCuts){
        printf("Finished. In total %lld tetrahedron rendered.\n", tetra_count);
    }else{
        printf("Finished!\n");
    }

}

