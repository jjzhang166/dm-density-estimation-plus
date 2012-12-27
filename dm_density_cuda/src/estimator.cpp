/*
 * estimator.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: lyang
 */
#include <vector>
#include <iostream>

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

void Estimater::computeDensity(){
	int loop_i;
	initialCUDA(tetrastream_->getTretras()->size(), gridmanager_->getSubGridSize());
	for(loop_i = 0; loop_i < gridmanager_->getSubGridNum(); loop_i ++){
		printf("Subblock number: %d\n", loop_i);
		tetrastream_->reset();
		gridmanager_->loadGrid(loop_i);
		//int count = 0;
		vector<Tetrahedron> * tetras_v = tetrastream_->getTretras();
		//Tetrahedron * tetras = &((*tetras_v)[0]);
		calculateGridWithCuda(tetras_v, gridmanager_);
		/*while(tetrastream_->hasnext()){
			int i=0, j=0, k=0;
			Tetrahedron tetra = *(tetrastream_->next());
			printf("Tetrahedron number: %d\n", count);
			count ++;
			if(tetra.maxx() - tetra.minx() >
				(gridmanager_->getEndPoint().x -
						gridmanager_->getStartPoint().x) / 2.0)
				continue;

			if(tetra.maxy() - tetra.miny() >
				(gridmanager_->getEndPoint().y -
						gridmanager_->getStartPoint().y) / 2.0)
				continue;

			if(tetra.maxz() - tetra.minz() >
				(gridmanager_->getEndPoint().z -
						gridmanager_->getStartPoint().z) / 2.0)
				continue;
			//printf("Tetra: %f %f %f\n", tetra.v1.x, tetra.v1.y, tetra.v1.z);
			REAL box = gridmanager_->getEndPoint().x -
					gridmanager_->getStartPoint().x;
			REAL ng = this->gridmanager_->getGridSize();

			REAL dx2 = box/ng/2;
			int sgs = gridmanager_->getSubGridSize();
			for(i = 0; i < sgs; i++){
				for(j = 0; j < sgs; j++){
					for(k = 0; k < sgs; k++){
						//calculate the actual coordinate
						Point p = gridmanager_->getPoint(i, j, k);
						p.x += dx2;
						p.y += dx2;
						p.z += dx2;
						if(tetra.isInTetra(p)){
							REAL cv = gridmanager_->getValue(i, j, k);
							gridmanager_ -> setValue(i, j, k, cv + 1/tetra.volume);
							//hasp = true;
						}
					}
				}
			}

		}
		*/
		gridmanager_->saveGrid();
	}
	finished_ = true;
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
	}
*/
}

bool Estimater::isFinished(){
	return finished_;
}

bool Estimater::isGood(){
	return good_;
}

