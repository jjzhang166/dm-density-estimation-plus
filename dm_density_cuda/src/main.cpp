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


//#include "grid.h"
#include "gridmanager.h"
#include "tetrahedron.h"
#include "tetrastream.h"
#include "estimator.h"


using namespace std;

namespace main_space{

    int gridsize    = 128;						//total grid size
    int subgridsize = 16;						//how many grid could be stored in the memory
    int inputmemgrid =16;						//the input memory grid size
    int gpu_mem_for_tetralist = 128*1024*1024;	//gpu memory for tetrahedron list
    string filename =  "E:\\multires_150";//"I:\\data\\MIP-00-00-00-run_050";		//the input data filename "E:\\multires_150";//
    string gridfilename = "I:\\sandbox\\tetrahedron.grid";	//output filename
    string velofilename = "I:\\sandbox\\tetrahedron.vgrid";	//velocity output filename
    bool isoutputres = false;					
    bool isVerbose = false;
    bool isInOrder = false;
    bool isVelocity = false;								//calculate velocity field?
    bool isSetBox = false;                       //set up a box for the grids
    Point setStartPoint;
    double boxsize = 32000.0;

void printUsage(string pname){
	fprintf(stderr, "Usage: %s \n %s \n %s \n %s\n %s\n %s\n %s \n %s \n %s \n %s \n %s\n %s\n"
            , pname.c_str()
			, "[-g <gridsize>]"
			, "[-s <subgridsize>]"
			, "[-df <datafilename>]"
			, "[-of <gridfilename>]"
			, "[-vfile <velocityfieldfilename> only applied when use -vel]"
			, "[-memgl <GPU memory for tetra list>]"
			, "[-t <numbers of tetra in memory>]"
			, "[-o] to output result in texts"
			, "[-order] if the data is in order"
			, "[-v] to show verbose"
			, "[-vel] if calculate velocity field"
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
			if(strcmp(args[k], "-g") == 0){
				ss << args[k + 1];
				ss >> gridsize;
			}else if(strcmp(args[k], "-s") == 0){
				ss << args[k + 1];
				ss >> subgridsize;
			}else if(strcmp(args[k], "-df") == 0){
				ss << args[k + 1];
				ss >> filename;
			}else if(strcmp(args[k], "-of") == 0){
				ss << args[k + 1];
				ss >> gridfilename;
			}else if(strcmp(args[k], "-vfile") == 0){
				ss << args[k + 1];
				ss >> velofilename;
			}else if(strcmp(args[k], "-memgl") == 0){
				ss << args[k + 1];
				ss >> gpu_mem_for_tetralist;
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
			}else if(strcmp(args[k], "-box") == 0){
				isSetBox = true;
                k++;
				ss << args[k];
				ss >> setStartPoint.x;
                k++;
                ss << args[k];
				ss >> setStartPoint.y;
                k++;
                ss << args[k];
				ss >> setStartPoint.z;
                k++;
                ss << args[k];
				ss >> boxsize;
                k--;
			}else{
				printUsage(args[0]);
				exit(1);
			}
			k += 2;
		}
	}
}

void normalizeVelocity(GridManager &grid, GridVelocityManager &gridvel){
	int i,j,k;

	int tgs = grid.getGridSize();
	for (i = 0; i < tgs; i++) {
		for (j = 0; j < tgs; j++) {
			for (k = 0; k < tgs; k++) {
				REAL den = grid.getValueByActualCoor(i, j, k);
				Point vel = gridvel.getValueByActualCoor(i, j, k);
				vel.x /= den;
				vel.y /= den;
				vel.z /= den;
				gridvel.setValueByActualCoor(i,j,k,vel);
			}
		}
	}
}

}

using namespace main_space;
int main(int argv, char * args[]){
	double io_t = 0, calc_t = 0, total_t = 0;
	timeval timediff;
	double t1, t2, t0 = 0;
	
	gettimeofday(&timediff, NULL);
	t0 = timediff.tv_sec + timediff.tv_usec / 1.0e6;

	readParameters(argv, args);
	printf("\n=========================DENSITY ESTIMATION==========================\n");
	printf("*****************************PARAMETERES*****************************\n");
	printf("Grid Size               = %d\n", gridsize);
	printf("Sub Grid Size           = %d\n", subgridsize);
	printf("Data File               = %s\n", filename.c_str());
	printf("Grid File               = %s\n", gridfilename.c_str());
	printf("Tetra in Mem            = %d\n", inputmemgrid);
	printf("GPU mem for tetra list  = %d\n", gpu_mem_for_tetralist);
    if(isSetBox){
        printf("Box                    = %f %f %f %f", setStartPoint.x, setStartPoint.y, setStartPoint.z, boxsize);
    }
	if(isVelocity){
		printf("Vel File               = %s\n", velofilename.c_str());
	}
	if(isInOrder){
		printf("The data is already in right order for speed up...\n");
	}


	printf("*********************************************************************\n");

	//tetrastream
	TetraStream tetraStream(filename, inputmemgrid, isVelocity);
	tetraStream.setIsInOrder(isInOrder);

	//compute the startpoint and endpoint
	Point startpoint;
	Point endpoint;
    gadget_header header = tetraStream.getHeader();
    if(!isSetBox){
        startpoint.x = 0;
        startpoint.y = 0;
        startpoint.z = 0;
        endpoint.x = (REAL)header.BoxSize;
        endpoint.y = (REAL)header.BoxSize;
        endpoint.z = (REAL)header.BoxSize;
    }else{
        startpoint.x = setStartPoint.x;
        startpoint.y = setStartPoint.y;
        startpoint.z = setStartPoint.z;
        endpoint.x = startpoint.x + boxsize;
        endpoint.y = startpoint.y + boxsize;
        endpoint.z = startpoint.z + boxsize;
    }

	//gridmanager
	GridManager grid(gridfilename, gridsize, subgridsize, 0, startpoint, endpoint);
	GridVelocityManager * gridvel = NULL;
	if(isVelocity){
	    //test	
        gridvel = new GridVelocityManager(velofilename, gridsize, subgridsize, startpoint, startpoint, endpoint);
	}

	//setup single vox_vol correction
	tetraStream.setCorrection(&grid);

	//estimator
	Estimater estimater(&tetraStream, &grid, gridvel, gpu_mem_for_tetralist);
	estimater.setVerbose(isVerbose);
	estimater.setIsVelocity(isVelocity);

	printf("*****************************COMPUTING ...***************************\n");
	//test remove computing
    estimater.computeDensity();
	estimater.getRunnintTime(io_t, calc_t);

	//normalize the velocity
	if(isVelocity){
		normalizeVelocity(grid, *gridvel);
	}

	//save to file
	gettimeofday(&timediff, NULL);
	t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;

	grid.saveToFile();
	if(isVelocity){
		//test
        gridvel->saveToFile();
		delete gridvel;
	}

	gettimeofday(&timediff, NULL);
	t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
	io_t += t2 - t1;

	if(estimater.isFinished()){
		printf("================================FINISHED=============================\n");
		//output as text file
		gettimeofday(&timediff, NULL);
	    t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		int i, j, k;
		if (isoutputres) {
			int tgs = grid.getGridSize();
			for (i = 0; i < tgs; i++) {
				for (j = 0; j < tgs; j++) {
					for (k = 0; k < tgs; k++) {
						REAL v = grid.getValueByActualCoor(k, j, i);
						if (v > 0) {
							printf("Ind: %d ==> %e\n", (k) + (j) * tgs + (i) * tgs * tgs, v);
						}
					}
				}
			}
		}

		gettimeofday(&timediff, NULL);
		t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		io_t += t2 - t1;
		total_t = t2 - t0;

		printf("Time: IO: %f sec, COMPUTING: %f sec, TOTAL: %f sec\n", io_t, calc_t, total_t);
		printf("=====================================================================\n");
		return 0;
	}else{
		printf("=================================ERROR===============================\n");
		exit(1);
	}
	
}
