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
#include "unistd.h"
#endif

#include "grids.h"
#include "tetrahedron.h"
#include "tetrastream.h"
#include "estimator.h"


using namespace std;

int gridsize    = 128;						//total grid size
int subgridsize = 16;						//how many grid could be stored in the memory
int inputmemgrid = 16;						//the input memory grid size
string filename = "E:\\multires_150";			//the input data filename
string gridfilename = "tetrahedron.grid";	//output filename
bool isoutputres = false;					

void printUsage(string pname){
	fprintf(stderr, "Usage: %s \n %s \n %s \n %s \n %s \n %s \n %s \n", pname.c_str()
			, "[-g <gridsize>]"
			, "[-s <subgridsize>]"
			, "[-df <datafilename>]"
			, "[-of <gridfilename>]"
			, "[-t <numbers of tetra in memory>]"
			, "[-o] to output result"
			);
}

void readParameters(int argv, char * args[]){
	int k = 1;
	if(argv == 1){
		return;
	}/*else if(argv % 2 ==0){
		printUsage(args[0]);
		exit(1);
	}*/else{
		while(k < argv){
			stringstream ss;
			//printf("fsafasdfa ---- %s\n", args[k]);
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
			}else if(strcmp(args[k], "-t") == 0){
				ss << args[k + 1];
				ss >> inputmemgrid;
			}else if(strcmp(args[k], "-o") == 0){
				isoutputres = true;
				k = k -1;
			}else{
				printUsage(args[0]);
				exit(1);
			}
			k += 2;
		}
	}
}

int main(int argv, char * args[]){
	double io_t = 0, calc_t = 0, total_t = 0;
	timeval timediff;
	double t1, t2, t0 = 0;
	
	gettimeofday(&timediff, NULL);
	t0 = timediff.tv_sec + timediff.tv_usec / 1.0e6;

	readParameters(argv, args);
	printf("\n=========================DENSITY ESTIMATION==========================\n");
	printf("*****************************PARAMETERES*****************************\n");
	printf("Grid Size     = %d\n", gridsize);
	printf("Sub Grid Size = %d\n", subgridsize);
	printf("Data File     = %s\n", filename.c_str());
	printf("Grid File     = %s\n", gridfilename.c_str());
	printf("Tetra in Mem  = %d\n", inputmemgrid);
	printf("*********************************************************************\n");

	TetraStream tetraStream(filename, inputmemgrid);
	GridManager grid(gridfilename, gridsize, subgridsize);
	Estimater estimater(&tetraStream, &grid);

	printf("*****************************COMPUTING ...***************************\n");

	estimater.computeDensity();
	estimater.getRunnintTime(io_t, calc_t);


	//single vex_vol correction
	REAL box = grid.getEndPoint().x - grid.getStartPoint().x;
	REAL ng = grid.getGridSize();
	REAL vox_vol = box * box * box / ng / ng / ng;
	int tetra_block_ind = 0;
	for(tetra_block_ind = 0; tetra_block_ind < tetraStream.getTotalBlockNum(); tetra_block_ind ++){
		tetraStream.loadBlock(tetra_block_ind);
		int tetra_ind = 0;
		for(tetra_ind = 0; tetra_ind < tetraStream.getBlockNumTetra(); tetra_ind ++){
			Tetrahedron & tetra = (tetraStream.getCurrentBlock())[tetra_ind];
			int xindmin = tetra.minx() / box * grid.getGridSize();
			int xindmax = tetra.maxx() / box * grid.getGridSize();
			int yindmin = tetra.miny() / box * grid.getGridSize();
			int yindmax = tetra.maxy() / box * grid.getGridSize();
			int zindmin = tetra.minz() / box * grid.getGridSize();
			int zindmax = tetra.maxz() / box * grid.getGridSize();
			int n_samples = (xindmax - xindmin + 1) * (yindmax - yindmin + 1) * (zindmax - zindmin + 1);
			if(n_samples == 1){
				grid.setValueByActualCoor(xindmin, yindmin, zindmin, 6.0 / vox_vol);
			}
		}
	}


	gettimeofday(&timediff, NULL);
	t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;

	grid.saveToFile();

	gettimeofday(&timediff, NULL);
	t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
	io_t += t2 - t1;



	if(estimater.isFinished()){
		printf("================================FINISHED=============================\n");
		gettimeofday(&timediff, NULL);
	    t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		int i, j, k;
		//isoutputres = true;
		if (isoutputres) {
			//grid.loadGrid(l);
			int tgs = grid.getGridSize();
			for (i = 0; i < tgs; i++) {
				for (j = 0; j < tgs; j++) {
					for (k = 0; k < tgs; k++) {
						double v = grid.getValueByActualCoor(k, j, i);
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

