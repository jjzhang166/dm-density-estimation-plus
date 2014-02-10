/*
 * estimator.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: lyang
 */
#include <vector>
#include <iostream>

using namespace std;

#include "tetrastream.h"
#include "grids.h"
#include "estimator.h"


Estimater::Estimater(TetraStream * tetrastream, GridManager * gridmanager){
	tetrastream_ = tetrastream;
	gridmanager_ = gridmanager;
	good_ = true;
	finished_ = false;
}

void Estimater::computeDensity(){
	//test
	/*Tetrahedron tetra;
	tetra.v1.x = 0;
	tetra.v1.y = 0;
	tetra.v1.z = 0;

	tetra.v2.x = 1;
	tetra.v2.y = 0;
	tetra.v2.z = 0;

	tetra.v3.x = 0;
	tetra.v3.y = 1;
	tetra.v3.z = 0;

	tetra.v4.x = 0;
	tetra.v4.y = 0;
	tetra.v4.z = 1;
	tetra.computeVolume();
	printf("Test in ");
	Point p0,p1, p2,p3;
	p0.x=1; p0.y = 1; p0.z = 1;
	p1.x=1; p1.y = 0; p1.z = 0;
	p2.x=0.1; p2.y =0.1; p2.z = 0.1;
	p3.x= 0.2; p3.y = 0.2; p3.z=0;
	std::cout << tetra.isInTetra(p0) << tetra.isInTetra(p1)
			<<tetra.isInTetra(p2) <<tetra.isInTetra(p3) <<endl;

	tetra.v1.x = 0.3;
	tetra.v1.y = 0.2;
	tetra.v1.z = 0.1;

	tetra.v2.x = 1.5;
	tetra.v2.y = 0.0;
	tetra.v2.z = 0.01;

	tetra.v3.x = 0.3;
	tetra.v3.y = 1;
	tetra.v3.z = 0.1;

	tetra.v4.x = 0;
	tetra.v4.y = 0;
	tetra.v4.z = 1.1;
	tetra.computeVolume();
	printf("Test volum: %f\n", tetra.volume);
	return;*/
	//printf("Sub Grid Numb: %d\n", gridmanager_->getSubGridNum());

	int loop_i;
	for(loop_i = 0; loop_i < gridmanager_->getSubGridNum(); loop_i ++){
		printf("Subblock number: %d\n", loop_i);
		tetrastream_->reset();
		gridmanager_->loadGrid(loop_i);
		int count = 0;
		while(tetrastream_->hasnext()){
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
			double box = gridmanager_->getEndPoint().x -
					gridmanager_->getStartPoint().x;
			double ng = this->gridmanager_->getGridSize();

			double dx2 = box/ng/2;
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
							double cv = gridmanager_->getValue(i, j, k);
							gridmanager_ -> setValue(i, j, k, cv + 1/tetra.volume);
							//hasp = true;
						}
					}
				}
			}

		}

		gridmanager_->saveGrid();
	}
	finished_ = true;

	//printf("Finished\n");

/*	int i, j, k, l;
	for(l = 0; l < gridmanager_->getSubGridNum(); l ++){
		gridmanager_->loadGrid(l);
		int gs = gridmanager_->getSubGridSize();
		for(i = 0; i < gs; i++){
			for(j = 0; j < gs; j++){
				for(k = 0; k < gs; k++){
					double v = gridmanager_->getValue(k, j, i);
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

