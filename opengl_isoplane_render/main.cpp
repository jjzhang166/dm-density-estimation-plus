/*
 * main.cpp
 *
 * Framework will be:
 * tetrahedron.h  			-- the tetrahedron structure
 * tetrastream.h			-- a stream that provides the tetrahedrons
 * grids.h					-- a scheme that defines the grids, provides sub-interface to the part of
 * 								grids structure. For example: change some portion of it. If the structure
 * 								is too large, then only the required portion is loaded, others are saved
 * 								in file. After commit, the current on-RAM portion will be also saved to
 * 								the memory.
 * estimator.h				-- takes TetraStream and GridManager as input, calculate the grids value
 *  Created on: Dec 17, 2012
 *      Author: lyang
 */
#include <cstdio>
#include <iostream>
#include <sstream>
#include <cstring>
#include <cmath>

#if defined(_WIN32) || defined(_WIN64)
#include "gettimeofday_win.h"
#else
#include <unistd.h>
#include <sys/time.h>
#endif

#ifdef __APPLE__
#include <GLUT/glut.h> // darwin uses glut.h rather than GL/glut.h
#else
#include <GL/glut.h>
#endif


//#include "grid.h"
#include "tetrahedron.h"
#include "indtetrastream.h"
#include "isoplane.h"
#include "render.h"


using namespace std;

namespace main_space{

    int inputmemgrid = 16;						//the input memory grid size
    string filename =  "E:\\multires_150";//"I:\\data\\MIP-00-00-00-run_050";		//the input data filename "E:\\multires_150";//
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

void printUsage(string pname){
	fprintf(stderr, "Usage: %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n %s\n"
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
	//double io_t = 0, calc_t = 0, total_t = 0;
	//timeval timediff;
	//double t1, t2, t0 = 0;
	
	//gettimeofday(&timediff, NULL);
	//t0 = timediff.tv_sec + timediff.tv_usec / 1.0e6;

	readParameters(argv, args);
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
	printf("Grid File               = %s\n", gridfilename.c_str());
	printf("Tetra in Mem            = %d\n", inputmemgrid);
    
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
            printf("    Without Memory Copying Mode -- but may be slower in regenerating tetras...");
        }

    }

    printf("*********************************************************************\n");

	IndTetraStream tetraStream(filename, inputmemgrid, parttype,
                               datagridsize, isVelocity, isHighMem, isAllData);
	tetraStream.setIsInOrder(isInOrder);
	tetraStream.setCorrection();
    
    //test
    /*TetraStreamer streamer(filename,
                           inputmemgrid,
                           parttype,
                           datagridsize,
                           isHighMem,
                           isAllData,
                           isVelocity,
                           true,
                           isInOrder);
    int count = 0;
    while(streamer.hasNext()){
        int nums;
        Tetrahedron * tetras;
        tetras = streamer.getNext(nums);
        //for(int i= 0; i < nums; i++){
        //    printf("%d %f     \n", i, tetras[i].volume);
        //    printf("%f %f %f\n",tetras[i].v1.x, tetras[i].v1.y, tetras[i].v1.z);
        //    printf("%f %f %f\n",tetras[i].v2.x, tetras[i].v2.y, tetras[i].v2.z);
        //    printf("%f %f %f\n",tetras[i].v3.x, tetras[i].v3.y, tetras[i].v3.z);
        //    printf("%f %f %f\n",tetras[i].v4.x, tetras[i].v4.y, tetras[i].v4.z);

        //}
        count += nums;
    }
    printf("%d\n", count);*/

    TetraIsoPlane isoplane(&tetraStream);
    //printf("IsoPlane ok\n");
	Render render(imagesize, boxsize, &isoplane, &argv, args);
    //printf("Main ok\n");
    //render.showPlane(1000);
    render.showPlane(26300);

    
    return 0;
}
