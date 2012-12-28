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

Estimater::Estimater(TetraStream * tetrastream, GridManager * gridmanager) {
	tetrastream_ = tetrastream;
	gridmanager_ = gridmanager;
	good_ = true;
	finished_ = false;

	box = gridmanager_->getEndPoint().x
				- gridmanager_->getStartPoint().x;
	ng = this->gridmanager_->getGridSize();
	dx2 = box / ng / 2;
}

//test whether this tetra is intouch with current cube
bool Estimater::testTouch(Tetrahedron * tetra){
	Point p1, p2, p3, p4;
	p1 = tetra->v1;
	p2 = tetra->v2;
	p3 = tetra->v3;
	p4 = tetra->v4;

	p1.x = p1.x / box * ng;
	p1.y = p1.y / box * ng;
	p1.z = p1.z / box * ng;

	if(gridmanager_->isInCurrentBlock(p1.x, p1.y, p1.z)){
		return true;
	}

	p2.x = p2.x / box * ng;
	p2.y = p2.y / box * ng;
	p2.z = p2.z / box * ng;
	if(gridmanager_->isInCurrentBlock(p2.x, p2.y, p2.z)){
		return true;
	}

	p3.x = p3.x / box * ng;
	p3.y = p3.y / box * ng;
	p3.z = p3.z / box * ng;
	if(gridmanager_->isInCurrentBlock(p3.x, p3.y, p3.z)){
		return true;
	}

	p4.x = p4.x / box * ng;
	p4.y = p4.y / box * ng;
	p4.z = p4.z / box * ng;
	if(gridmanager_->isInCurrentBlock(p4.x, p4.y, p4.z)){
		return true;
	}

	if(tetra->isInTetra(gridmanager_->getPoint(0, 0, 0)))
		return true;

	int sg = gridmanager_->getSubGridSize();
	if(tetra->isInTetra(gridmanager_->getPoint(0, 0, sg)))
		return true;

	if(tetra->isInTetra(gridmanager_->getPoint(0, sg, 0)))
		return true;

	if(tetra->isInTetra(gridmanager_->getPoint(0, sg, sg)))
		return true;

	if(tetra->isInTetra(gridmanager_->getPoint(sg, 0, 0)))
		return true;

	if(tetra->isInTetra(gridmanager_->getPoint(sg, 0, sg)))
		return true;

	if(tetra->isInTetra(gridmanager_->getPoint(sg, sg, 0)))
		return true;

	if(tetra->isInTetra(gridmanager_->getPoint(sg, sg, sg)))
		return true;

	return false;
}

void Estimater::computeDensity() {
	//test
	int loop_i;


	for (loop_i = 0; loop_i < gridmanager_->getSubGridNum(); loop_i++) {
		printf("Subblock number: %d\n", loop_i);
		tetrastream_->reset();
		gridmanager_->loadGrid(loop_i);
		int count = 0;
		while (tetrastream_->hasnext() && count < 20) {
			int i = 0, j = 0, k = 0;
			Tetrahedron tetra = *(tetrastream_->next());
			count++;
			if(!testTouch(&tetra)){
				continue;
			}
			printf("Tetrahedron number: %d\n", count);
			if (tetra.maxx() - tetra.minx()
					> (gridmanager_->getEndPoint().x
							- gridmanager_->getStartPoint().x) / 2.0)
				continue;

			if (tetra.maxy() - tetra.miny()
					> (gridmanager_->getEndPoint().y
							- gridmanager_->getStartPoint().y) / 2.0)
				continue;

			if (tetra.maxz() - tetra.minz()
					> (gridmanager_->getEndPoint().z
							- gridmanager_->getStartPoint().z) / 2.0)
				continue;
			//printf("Tetra: %f %f %f\n", tetra.v1.x, tetra.v1.y, tetra.v1.z);


			int sgs = gridmanager_->getSubGridSize();
			for (i = 0; i < sgs; i++) {
				for (j = 0; j < sgs; j++) {
					for (k = 0; k < sgs; k++) {
						//calculate the actual coordinate
						Point p = gridmanager_->getPoint(i, j, k);
						p.x += dx2;
						p.y += dx2;
						p.z += dx2;
						if (tetra.isInTetra(p)) {
							double cv = gridmanager_->getValue(i, j, k);
							gridmanager_->setValue(i, j, k,
									cv + 1 / tetra.volume);
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

	/*int i, j, k, l;
	for (l = 0; l < gridmanager_->getSubGridNum(); l++) {
		gridmanager_->loadGrid(l);
		int gs = gridmanager_->getSubGridSize();
		for (i = 0; i < gs; i++) {
			for (j = 0; j < gs; j++) {
				for (k = 0; k < gs; k++) {
					double v = gridmanager_->getValue(k, j, i);
					//printf("%f\n", v );
					if (v > 0) {
						printf("Ind: %d ==> %e\n", k + j * gs + i * gs * gs, v);
					}
				}
			}
		}
	}*/

}

bool Estimater::isFinished() {
	return finished_;
}

bool Estimater::isGood() {
	return good_;
}

