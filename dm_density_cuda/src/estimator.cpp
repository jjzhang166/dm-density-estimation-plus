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
#endif

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#include "types.h"
#include "tetrastream.h"
#include "grids.h"
#include "estimator.h"
#include "kernel.h"

Estimater::Estimater(TetraStream * tetrastream, GridManager * gridmanager){
	tetrastream_ = tetrastream;
	gridmanager_ = gridmanager;
	good_ = true;
	finished_ = false;
}

void Estimater::getRunnintTime(double &iotime, double &calctime){
	iotime += this->iotime_;
	calctime += this->calctime_;
}

void Estimater::computeDensity(){
	timeval timediff;
	double t1, t2 = 0;
	iotime_ = 0;
	calctime_ = 0;

	int loop_i;
	if(initialCUDA(tetrastream_, gridmanager_) != cudaSuccess){
		exit(1);
	}

	int tetra_ind = 0;
	int tetra_num_block = tetrastream_->getTotalBlockNum();
	for(tetra_ind = 0; tetra_ind < tetra_num_block; tetra_ind ++){
		printf("TetraBlocks: %d/%d\n", tetra_ind + 1, tetra_num_block);

		gettimeofday(&timediff, NULL);
	    t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		tetrastream_->loadBlock(tetra_ind);
		gettimeofday(&timediff, NULL);
		t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		iotime_ += t2 - t1;

		gettimeofday(&timediff, NULL);
	    t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		if(computeTetraMemWithCuda() != cudaSuccess)
			exit(1);
		gettimeofday(&timediff, NULL);
		t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
		calctime_ += t2 - t1;

		//computeTetraSelectionWithCuda();
		printf("=========[---10---20---30---40---50---60---70---80---90--100-]========\n");
		printf("=========[");
		int res_print_ = gridmanager_->getSubGridNum() / 50;
		if(res_print_ == 0){
			res_print_ = 1;
		}

		for(loop_i = 0; loop_i < gridmanager_->getSubGridNum(); loop_i ++){

			gettimeofday(&timediff, NULL);
			t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
			if((loop_i + 1) % (res_print_) == 0){
				//printf(">");
				cout<<"<";
				cout.flush();
			}
			gridmanager_->loadGrid(loop_i);
			gettimeofday(&timediff, NULL);
			t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
			iotime_ += t2 - t1;

			//int count = 0;
			gettimeofday(&timediff, NULL);
			t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
			calculateGridWithCuda();
			gettimeofday(&timediff, NULL);
			t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
			calctime_ += t2 - t1;

			gettimeofday(&timediff, NULL);
			t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
			gridmanager_->saveGrid();
			gettimeofday(&timediff, NULL);
			t2 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
			iotime_ += t2 - t1;
		}
		finished_ = true;
		printf("]========\n");
	}
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

