/*********************************************************************/
/* get the density use the iso-z-cutter.                             */
/*this is very fast, no need to calculate the interpolation every time*/
/*Author: Lin F. Yang                                                */
/*********************************************************************/

#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>
#include <cstring>
#include <cmath>

#if defined(_WIN32) || defined(_WIN64)
//#include "gettimeofday_win.h"
#else
#include <unistd.h>
#include <sys/time.h>
#endif


//#include "grid.h"
#include "tetrahedron.h"
#include "indtetrastream.h"
#include "isoplane.h"
#include "render.h"
#include "denrender.h"


namespace main_space{
    
    int inputmemgrid = 16;						//the input memory grid size
    string filename =  "I:\\data\\32Mpc_050_S4";//"I:\\data\\MIP-00-00-00-run_050";		//the input data filename "E:\\multires_150";//
    
    float * colorimage;
    
    //GLuint textureIni;
    
    string gridfilename = "I:\\sandbox\\tetrahedron.grid";	//output filename
    string velofilename = "I:\\sandbox\\tetrahedron.vgrid";	//velocity output filename
    bool isoutputres = false;
    bool isVerbose = false;
    bool isInOrder = false;
    bool isVelocity = false;								//calculate velocity field?
    
    //load all the particles into memory?
    bool isHighMem = true;
    //return the all particle pointer?
    bool isAllData = false;
    
    //if use -1, then use the particle gridsize as the gridsize
    //otherwise use the user setting
    int datagridsize = -1;
    //the particle type in the gadget file
    int parttype = 1;
    
    bool isSetBox = false;                       //set up a box for the grids
    Point setStartPoint;
    double boxsize = 32000.0;
    int imagesize = 512;
    int numOfCuts = 0;
    float dz = 0;
    float startz = 0;
    
    void printUsage(string pname){
        fprintf(stderr, "Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n"
                , pname.c_str()
                , "[-imsize <imagesize>]"
                , "[-df <datafilename>]"
                , "[-of <gridfilename>]"
                , "[-vfile <velocityfieldfilename> only applied when use -vel]"
                , "[-t <numbers of tetra in memory>]"
                , "[-o] to output result in texts"
                , "[-order] if the data is in order"
                , "[-v] to show verbose"
                , "[-vel] if calculate velocity field"
                , "[-dgridsize] default: -1 to use the npart^(1/3) as gridsize"
                , "[-parttype] default: 1. Use 0-NTYPE data in the gadgetfile"
                , "[-lowmem] use low memory mode (don't load all part in mem)"
                , "[-nalldata] only usable in highmem mode"
                , "[-startz] the starting z-coordinates to calculate the cuts"
                , "[-dz] the interval between each 2 z-plane"
                , "[-numz] the number of z-planes"
                , "[-box <x0> <y0> <z0> <boxsize>] setup the start point, and the boxsize. The box should be inside the data's box, otherwise some unpredictable side effects will comes out"
                );
    }
    
    void readParameters(int argv, char * args[]){
        int k = 1;
        if(argv == 1){
            return;
        }else{
            while(k < argv){
                stringstream ss;
                if(strcmp(args[k], "-imsize") == 0){
                    ss << args[k + 1];
                    ss >> imagesize;
                }else if(strcmp(args[k], "-df") == 0){
                    ss << args[k + 1];
                    ss >> filename;
                }else if(strcmp(args[k], "-of") == 0){
                    ss << args[k + 1];
                    ss >> gridfilename;
                }else if(strcmp(args[k], "-vfile") == 0){
                    ss << args[k + 1];
                    ss >> velofilename;
                }else if(strcmp(args[k], "-t") == 0){
                    ss << args[k + 1];
                    ss >> inputmemgrid;
                }else if(strcmp(args[k], "-o") == 0){
                    isoutputres = true;
                    k = k -1;
                }else if(strcmp(args[k], "-v") == 0){
                    isVerbose = true;
                    k = k -1;
                }else if(strcmp(args[k], "-order") == 0){
                    isInOrder = true;
                    k = k -1;
                }else if(strcmp(args[k], "-vel") == 0){
                    isVelocity = true;
                    k = k -1;
                }else if(strcmp(args[k], "-dgridsize") == 0){
                    ss << args[k + 1];
                    ss >> datagridsize;
                }else if(strcmp(args[k], "-parttype") == 0){
                    ss << args[k + 1];
                    ss >> parttype;
                }else if(strcmp(args[k], "-lowmem") == 0){
                    isHighMem = false;
                    k = k -1;
                }else if(strcmp(args[k], "-alldata") == 0){
                    isAllData = true;
                    k = k -1;
                }else if(strcmp(args[k], "-startz") == 0){
                    ss << args[k + 1];
                    ss >> startz;
                }else if(strcmp(args[k], "-dz") == 0){
                    ss << args[k + 1];
                    ss >> dz;
                }else if(strcmp(args[k], "-numz") == 0){
                    ss << args[k + 1];
                    ss >> numOfCuts;
                }else if(strcmp(args[k], "-box") == 0){
                    isSetBox = true;
                    k++;
                    ss << args[k] << " ";
                    ss << args[k + 1] << " ";
                    ss << args[k + 2] << " ";
                    ss << args[k + 3] << " ";
                    ss >> setStartPoint.x;
                    ss >> setStartPoint.y;
                    ss >> setStartPoint.z;
                    ss >> boxsize;
                    k += 2;
                }else{
                    printUsage(args[0]);
                    exit(1);
                }
                k += 2;
            }
        }
    }
    
}



using namespace main_space;

int main(int argv, char * args[]){
    
	readParameters(argv, args);
    if(numOfCuts == 0){
        numOfCuts = imagesize;
    }
    
    //test
    TetraStreamer streamer(filename,
                           inputmemgrid,
                           parttype,
                           datagridsize,
                           isHighMem,
                           isAllData,
                           isVelocity,
                           true,
                           isInOrder);
    
    boxsize = streamer.getIndTetraStream()->getHeader().BoxSize;
    dz = boxsize / numOfCuts;
    if(datagridsize == -1){
        datagridsize = ceil(pow(streamer.getIndTetraStream()->getHeader().npart[parttype], 1.0 / 3.0));
    }
    
    
    printf("\n=========================DENSITY ESTIMATION==========================\n");
	printf("*****************************PARAMETERES*****************************\n");
    printf("Render Image Size       = %d\n", imagesize);
	printf("Data File               = %s\n", filename.c_str());
    if(datagridsize == -1){
        printf("DataGridsize            = [to be determined by data]\n");
    }else{
        printf("DataGridsize            = %d\n", datagridsize);
    }
    printf("Particle Type           = %d\n", parttype);
	printf("Output File             = %s\n", gridfilename.c_str());
	printf("Tetra in Mem            = %d\n", inputmemgrid);
    printf("Rendering %d z-cuts of the density field. \nStart from z = %f, with dz = %f\n", numOfCuts, startz, dz);
    
    if(isSetBox){
        printf("Box                    = %f %f %f %f\n",
               setStartPoint.x, setStartPoint.y, setStartPoint.z, boxsize);
    }
	if(isVelocity){
		printf("Vel File               = %s\n", velofilename.c_str());
	}
	if(isInOrder){
		printf("The data is already in right order for speed up...\n");
	}
    if(!isHighMem){
        printf("Low Memory mode: slower in reading file...\n");
    }else{
        printf("Block Memory Operation:\n");
        if(!isAllData){
            printf("    Use Memory Copy Mode -- but may be faster without regenerating the tetras...\n");
        }else{
            printf("    Without Memory Copying Mode -- but may be slower in regenerating tetras...\n");
        }
        
    }
    
    printf("*********************************************************************\n");
    
    
    
/*    //initiate openGL
    glutInit(&argv, args);
    glutInitDisplayMode(GLUT_RGBA | GLUT_SINGLE);
    glutInitWindowSize(imagesize, imagesize);
    glutCreateWindow("Dark Matter Density rendering!");
    
#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    glewInit();
#endif*/
    
    DenRender render(imagesize, boxsize,
                     startz, dz, numOfCuts,
                     &argv, args);
    
    int count = 0;
    
    //render

    int tcount = datagridsize * datagridsize * datagridsize * 6 / 10;
    
    printf("Start rendering ...\n");
    while(streamer.hasNext()){
        int nums;
        Tetrahedron * tetras;
        tetras = streamer.getNext(nums);
        for(int i= 0; i < nums; i++){
            render.rend(tetras[i]);
            if((count %  tcount )== 0){
                printf(">");
                cout.flush();
            }
            count ++;
        }
    }
    render.finish();
    float * im = render.getDenfield();
    
    printf("\nFinished. In total %d tetrahedron rendered.\nSaving ...\n", count);
    
    //head used 256 bytes
    //the first is imagesize
    //the second the numOfCuts
    //the third is a float number boxsize
    //the 4-th is a float number startz
    //the 5-th is a fload number dz
    //All others are 0
    int head[59];
    fstream outstream;
    outstream.open(gridfilename.c_str(), ios::out | ios::binary);
    while(!outstream.good()){
        printf("Output error, calculation not saved...!\n");
        printf("Input new filename:\n");
        cin >> gridfilename;
        outstream.clear();
        outstream.open(gridfilename.c_str(), ios::out | ios::binary);
    }
    outstream.write((char *) &imagesize, sizeof(int));
    outstream.write((char *) &numOfCuts, sizeof(int));
    outstream.write((char *) &boxsize, sizeof(float));
    outstream.write((char *) &startz, sizeof(float));
    outstream.write((char *) &dz, sizeof(float));
    outstream.write((char *) head, sizeof(int) * 59);
    outstream.write((char *) im, sizeof(float) * imagesize * imagesize * numOfCuts);
    outstream.close();
    
    return 0;
}

