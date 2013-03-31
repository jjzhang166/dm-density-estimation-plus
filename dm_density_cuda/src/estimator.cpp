/*
 * estimator.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: lyang
 */
#include <vector>
#include <iostream>

#if defined(_WIN32) || defined(_WIN64)
#include "gettimeofday_win.h"
#else
#include "unistd.h"
#include <sys/time.h>
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#include "types.h"
#include "indtetrastream.h"
#include "gridmanager.h"
#include "estimator.h"
#include "kernel.h"
#include "processbar.h"


void Estimater::initialize(TetraStreamer * tetrastream, GridManager * gridmanager){
	tetrastream_ = tetrastream;
	gridmanager_ = gridmanager;
	good_ = true;
	finished_ = false;
	gpu_tetra_list_mem_lim =  128*1024*1024;		//128M
	isVerbose_ = false;
	isVelocity_ = false;
}

Estimater::Estimater(TetraStreamer * tetrastream, GridManager * gridmanager){
	initialize(tetrastream, gridmanager);
}

Estimater::Estimater(TetraStreamer * tetrastream, GridManager * gridmanager, int tetra_list_mem_lim){
	initialize(tetrastream, gridmanager);
	gpu_tetra_list_mem_lim = tetra_list_mem_lim;
}

Estimater::Estimater(TetraStreamer * tetrastream, GridManager * gridmanager,  GridVelocityManager * gridvelocity, int tetra_list_mem_lim){
	initialize(tetrastream, gridmanager);
	gridvelocity_ = gridvelocity;
	gpu_tetra_list_mem_lim = tetra_list_mem_lim;
}


void Estimater::getRunnintTime(double &iotime, double &calctime){
	iotime += this->iotime_;
	calctime += this->calctime_;
}


void Estimater::setVerbose(bool verbose){
	isVerbose_ = verbose;
}

void Estimater::computeDensity(){
	timeval timediff;
	double t1, t2 = 0;
	iotime_ = 0;
	calctime_ = 0;
	finished_ = false;
	//int pbar_type = 0;

	//testing
	//isVerbose_ = true;

	//if(isVerbose_){
	//	pbar_type = 1;
	//}else{
	//	pbar_type = 0;
	//}
	//ProcessBar process(tetrastream_->getTotalBlockNum() * gridmanager_-> getSubGridNum(), pbar_type);

	int loop_i;

	if(isVerbose_)
		printf("Initialing CUDA devices ...\n");

	if(initialCUDA(tetrastream_->getTetraContLimit(),
                   gridmanager_,
                   gpu_tetra_list_mem_lim,
                   gridvelocity_,
                   isVelocity_) != cudaSuccess){
		return;
	}

	//int tetra_ind = 0;
	//int tetra_num_block = tetrastream_->getTotalBlockNum();
	//process.start();

	//for(tetra_ind = 0; tetra_ind < tetra_num_block; tetra_ind ++){
    while(tetrastream_->hasNext()){
		//if(isVerbose_)
		//	printf("Loading TetraBlocks: %d/%d\n", tetra_ind + 1, tetra_num_block);

        int num_tetra_;
        Tetrahedron * tetras_;
		gettimeofday(&timediff, NULL);
	    t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		//tetrastream_->loadBlock(tetra_ind);
        tetras_ = tetrastream_->getNext(num_tetra_);
		gettimeofday(&timediff, NULL);
		t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		iotime_ += t2 - t1;
		
		//if(isVerbose_)
		//	printf("LoadedTetraBlocks: %d/%d, takes time %f secs\n", tetra_ind + 1, tetra_num_block, t2 - t1);

		if(isVerbose_)
			printf("Computing how many tetra memory need for GPU ... ");

		gettimeofday(&timediff, NULL);
	    t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		if(computeTetraMemWithCuda(tetras_, num_tetra_) != cudaSuccess)
			return;
		gettimeofday(&timediff, NULL);
		t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		calctime_ += t2 - t1;
		
		if(isVerbose_)
			printf("Costs %f secs\n", t2 - t1);


		bool hasnext = true;
		while(hasnext){
			if(isVerbose_)
				printf("Computing the tetrahedron list for each sub-grid block...\n");
			gettimeofday(&timediff, NULL);
			t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
			if(computeTetraSelectionWithCuda(hasnext)!=cudaSuccess){
				return;
			}
			gettimeofday(&timediff, NULL);
			t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
			calctime_ += t2 - t1;

			if(isVerbose_){
				if(hasnext){
					printf("Cost %f secs. GPU memory insufficient, divided to multiple step.\n", t2 - t1);
				}else{
					printf("Cost %f secs.\n", t2 - t1);
				}
			}
						
			int res_print_ = gridmanager_->getSubGridNum() / 50;
			if(res_print_ == 0){
				res_print_ = 1;
			}

			if(isVerbose_){
				printf("Looping over the grids... \n");
			}

			gettimeofday(&timediff, NULL);
			t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;

			for(loop_i = 0; loop_i < gridmanager_-> getSubGridNum(); loop_i ++){
				//process.setvalue(loop_i + tetra_ind * (gridmanager_-> getSubGridNum()));
				gridmanager_->loadGrid(loop_i);
				if(isVelocity_){
					gridvelocity_->loadGrid(loop_i);
				}
				calculateGridWithCuda();
			}

			gettimeofday(&timediff, NULL);
			t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
			calctime_ += t2 - t1;
			if(isVerbose_)
				printf("Costs %f secs\n", t2 - t1);
		}

	}
	finished_ = true;
	//process.end();
	finishCUDA();

	//printf("Finished\n");

/*	int i, j, k, l;
	for(l = 0; l < gridmanager_->getSubGridNum(); l ++){
		gridmanager_->loadGrid(l);
		int gs = gridmanager_->getSubGridSize();
		for(i = 0; i < gs; i++){
			for(j = 0; j < gs; j++){
				for(k = 0; k < gs; k++){
					REAL v = gridmanager_->getValue(k, j, i);
					//printf("%f\n", v );
					if(v > 0){
						printf("Ind: %d ==> %e\n", k + j * gs + i * gs * gs, v);
					}
				}
			}
		}
	}*/

}

bool Estimater::isFinished(){
	return finished_;
}

bool Estimater::isGood(){
	return good_;
}

void Estimater::setIsVelocity(bool isvel){
	isVelocity_ = isvel;
}
