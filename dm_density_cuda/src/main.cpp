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

#include "grids.h"
#include "tetrahedron.h"
#include "tetrastream.h"
#include "estimator.h"


using namespace std;

int gridsize    = 128;						//total grid size
int subgridsize = 16;						//how many grid could be stored in the memory
int inputmemgrid = 16;						//the input memory grid size
string filename = "E:\\multires_150";			//the input data filename
string gridfilename = "tetrahegen.grid";	//output filename
bool isoutputres = false;					

void printUsage(string pname){
	printf("Usage: %s \n %s \n %s \n %s \n %s \n %s \n %s \n", pname.c_str()
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
	//int loop_i;
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


	//test grid

/*	grid.setValueByActualCoor(130,110,100, 8.9889);
	grid.saveGrid();
	if(grid.loadGrid(6)){
		grid.setValue(1,2,3, 3.1415926);
		grid.saveGrid();
		printf("Current grid: %d\n", grid.getCurrentInd());
	}
	if(grid.loadGrid(1, 0, 1)){
		printf("Current grid: %d\n", grid.getCurrentInd());
	}
	if(grid.loadGrid(6)){
		printf("GridTesting value: %f\n", grid.getValue(1,2,3));
	}
	printf("Testing value %f\n", grid.getValueByActualCoor(130, 110, 100));

	int ai = 129;
	int aj = 210;
	int ak = 101;
	int ci, cj, ck;
	grid.loadGridByActualCoor(ai, aj, ak);
	//printf("A ind -> %d\n", grid.getCurrentInd());
	grid.actual2Current(ai, aj, ak, ci, cj, ck);
	grid.current2Actual(ci, cj, ck, ai, aj, ak);
	printf("Testing Converting %d %d %d %d %d %d\n",
			ai, aj, ak, ci, cj, ck);
*/
	//test gadget reader
	/*GadgetReader::GSnap gsnap(filename);
	std::cout << gsnap.GetFileName() << endl;
	std::cout << gsnap.GetFormat() << endl;
	std::cout << gsnap.GetHeader(0).BoxSize << endl;
	std::cout << gsnap.GetHeader(0).Omega0 << endl;
	std::cout << gsnap.GetHeader(0).OmegaLambda << endl;
	std::cout << gsnap.GetHeader(0).npart[gsnap.GetFormat()] << endl;
	std::cout << pow(gsnap.GetNpart(gsnap.GetFormat()), 1.0/3.0) << endl;
	set<string> bs = gsnap.GetBlocks();
	vector<string> bsv;
	std::copy(bs.begin(), bs.end(), std::back_inserter(bsv));

	for(std::vector<string>::iterator it=bsv.begin(); it!=bsv.end(); ++it){
		cout << *it << endl;
	}

	int nparts = gsnap.GetNpart(gsnap.GetFormat());

	float * data_array = new float[nparts * 3];
	float * sx = new float[nparts];
	float * sy = new float[nparts];
	float * sz = new float[nparts];
	int * ids = new int[nparts];
	gsnap.GetBlock("POS ", data_array, nparts, 0, 0);
	gsnap.GetBlock("ID  ", ids, nparts, 0, 0);

	//sorting
	for(int i = 0; i < nparts; i++){
		sx[ids[i]] = data_array[i * 3];
		sy[ids[i]] = data_array[i * 3 + 1];
		sz[ids[i]] = data_array[i * 3 + 2];
	}
	delete data_array;

	for(int i = 0; i < nparts; i++){
		printf("%6f %6f %6f\n", sx[i], sy[i], sz[i]);
	}
	delete ids;
	delete sx;
	delete sy;
	delete sz;
	*/
	printf("*****************************COMPUTING ...***************************\n");

	estimater.computeDensity();

	if(estimater.isFinished()){
		printf("================================FINISHED=============================\n");
		int i, j, k, l;
		for (l = 0; l < grid.getSubGridNum() && isoutputres; l++) {
			grid.loadGrid(l);
			int gs = grid.getSubGridSize();
			int tgs = grid.getGridSize();
			for (i = 0; i < gs; i++) {
				for (j = 0; j < gs; j++) {
					for (k = 0; k < gs; k++) {
						double v = grid.getValue(k, j, i);
						if (v > 0) {
							int ng = grid.getGridSize()/grid.getSubGridSize();

							int k0 = l % ng * grid.getSubGridSize();
							int j0 = (l / ng) % ng * grid.getSubGridSize();
							int i0 = (l / ng / ng) % ng * grid.getSubGridSize();
							printf("Ind: %d ==> %e\n", (k0+k) + (j0+j) * tgs + (i0+i) * tgs * tgs,
									v);
						}
					}
				}
			}
		}
			
		return 0;
	}else{
		printf("=================================ERROR===============================\n");
		exit(1);
	}
}

