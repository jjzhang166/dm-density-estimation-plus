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
//#include "tetraselector.h"
#include "estimator.h"

Estimater::Estimater(TetraStream * tetrastream, GridManager * gridmanager) {
	tetrastream_ = tetrastream;
	gridmanager_ = gridmanager;
	//tetraselector_ = new TetraSelector(gridmanager_, tetrastream_);
	good_ = true;
	finished_ = false;

	box = gridmanager_->getEndPoint().x
				- gridmanager_->getStartPoint().x;
	ng = this->gridmanager_->getGridSize();
	dx2 = box / ng / 2;
}

Estimater::~Estimater(){
	//delete tetraselector_;
}

bool isInBox(Point &p, Point &v1, Point &v2, double dx2){
	return (p.x >= v1.x - dx2 && p.x < v2.x + dx2) && (p.y >= v1.y - dx2 && p.y < v2.y + dx2)
			&& (p.z >= v1.z - dx2 && p.z < v2.z + dx2);
}

//test whether this tetra is intouch with current cube
bool Estimater::testTouch(Tetrahedron * tetra){
	Point p1, p2, p3, p4, p5, p6, p7, p8;
	Point v1, v2, v3, v4, v5, v6, v7, v8;
	int sg = gridmanager_->getSubGridSize();

	v1 = gridmanager_->getPoint(0, 0, 0);
	/*v2 = gridmanager_->getPoint(0, 0,sg);
	v3 = gridmanager_->getPoint(0, sg,0);
	v4 = gridmanager_->getPoint(0,sg,sg);
	v5 = gridmanager_->getPoint(sg,0, 0);
	v6 = gridmanager_->getPoint(sg,0,sg);
	v7 = gridmanager_->getPoint(sg,sg,0);*/
	v8 = gridmanager_->getPoint(sg,sg,sg);


	p1 = tetra->v1;
	p2 = tetra->v2;
	p3 = tetra->v3;
	p4 = tetra->v4;

	double minx = min(min(min(p1.x, p2.x), p3.x), p4.x);
	double maxx = max(max(max(p1.x, p2.x), p3.x), p4.x);
	double miny = min(min(min(p1.y, p2.y), p3.y), p4.y);
	double maxy = max(max(max(p1.y, p2.y), p3.y), p4.y);
	double minz = min(min(min(p1.z, p2.z), p3.z), p4.z);
	double maxz = max(max(max(p1.z, p2.z), p3.z), p4.z);

	if (minx > v8.x - 2*dx2 || maxx < v1.x + 2*dx2
		|| miny > v8.y - 2*dx2 || maxy < v1.y + 2*dx2
		|| minz > v8.z - 2*dx2 || maxz < v1.z + 2*dx2){
		return false;
	}
	return true;

	/*p1.x = minx;
	p1.y = miny;
	p1.z = minz;

	p2.x = minx;
	p2.y = miny;
	p2.z = maxz;

	p3.x = minx;
	p3.y = maxy;
	p3.z = minz;

	p4.x = minx;
	p4.y = maxy;
	p4.z = maxz;

	p5.x = maxx;
	p5.y = miny;
	p5.z = minz;

	p6.x = maxx;
	p6.y = miny;
	p6.z = maxz;

	p7.x = maxx;
	p7.y = maxy;
	p7.z = minz;

	p8.x = maxx;
	p8.y = maxy;
	p8.z = maxz;


	if(isInBox(p1, v1, v8, 2*dx2)
	|| isInBox(p2, v1, v8, 2*dx2)
	|| isInBox(p3, v1, v8, 2*dx2)
	|| isInBox(p4, v1, v8, 2*dx2)
	|| isInBox(p5, v1, v8, 2*dx2)
	|| isInBox(p6, v1, v8, 2*dx2)
	|| isInBox(p7, v1, v8, 2*dx2)
	|| isInBox(p8, v1, v8, 2*dx2)){
		return true;
	}

	//check whether the edge intersects with the cube
	if(minx <= v1.x && maxx >= v8.x){
		if(miny <= v8.y && miny >= v1.y && minz <= v8.z && minz >= v1.z)
			return true;
		if(miny <= v8.y && miny >= v1.y && maxz <= v8.z && maxz >= v1.z)
			return true;
		if(maxy <= v8.y && maxy >= v1.y && minz <= v8.z && minz >= v1.z)
			return true;
		if(maxy <= v8.y && maxy >= v1.y && maxz <= v8.z && maxz >= v1.z)
			return true;
	}

	if(miny <= v1.y && maxy >= v8.y){
		if(minx <= v8.x && minx >= v1.x && minz <= v8.z && minz >= v1.z)
			return true;
		if(minx <= v8.x && minx >= v1.x && maxz <= v8.z && maxz >= v1.z)
			return true;
		if(maxx <= v8.y && maxx >= v1.x && minz <= v8.z && minz >= v1.z)
			return true;
		if(maxx <= v8.x && maxx >= v1.x && maxz <= v8.z && maxz >= v1.z)
			return true;
	}

	if(minz <= v1.z && maxz >= v8.z){
		if(minx <= v8.x && minx >= v1.x && miny <= v8.y && miny >= v1.y)
			return true;
		if(minx <= v8.x && minx >= v1.x && maxy <= v8.y && maxy >= v1.y)
			return true;
		if(maxx <= v8.y && maxx >= v1.x && miny <= v8.y && miny >= v1.y)
			return true;
		if(maxx <= v8.x && maxx >= v1.x && maxy <= v8.y && maxy >= v1.y)
			return true;
	}*/

	/*if(tetra->isInTetra(v1)
	|| tetra->isInTetra(v2)
	|| tetra->isInTetra(v3)
	|| tetra->isInTetra(v4)
	|| tetra->isInTetra(v5)
	|| tetra->isInTetra(v6)
	|| tetra->isInTetra(v7)
	|| tetra->isInTetra(v8))
		return true;
	*/
	//return false;
}

void Estimater::computeDensity() {
	//test
	int loop_i;

	//printf("================================FINISHED=============================\n");
	printf("=========[---10---20---30---40---50---60---70---80---90--100]========\n");
	printf("=========[");
	int res_print_ = gridmanager_->getSubGridNum() / 50;
	if(res_print_ == 0){
		res_print_ = 1;
	}
	for (loop_i = 0; loop_i < gridmanager_->getSubGridNum(); loop_i++) {
		//printf("%d  ", loop_i);
		if((loop_i + 1) % (res_print_) == 0){
			printf(">");
		}
		std::cout.flush();
		tetrastream_->reset();
		gridmanager_->loadGrid(loop_i);
		int count = 0;
		//int count_tetra = 0;
		while (tetrastream_->hasnext()) {
			int i = 0, j = 0, k = 0;
			Tetrahedron tetra = *(tetrastream_->next());
			count++;
			if(!testTouch(&tetra)){
				continue;
			}

			//count_tetra ++;

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
			//printf("Tetrahedron number: %d\n", count);
			bool hasp = false;

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
							hasp = true;
						}
					}
				}
			}
//			if(hasp){
//				count_tetra ++;
//			}

		}

		//printf("%d ==> %d\n", loop_i, count_tetra);
		gridmanager_->saveGrid();
	}
	printf("]========\n");
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


/*
 __device__ bool isInTouch(int ind, int subgs, int gs, int nsg, float box, float dx2, Tetrahedron * tetra){

	Point p1, p2, p3, p4, p5, p6, p7, p8;
	Point v1, v8;
	int sg = nsg;

	v1 = getPoint(ind, 0, 0, 0, subgs, gs, nsg, box);
	v8 = getPoint(ind, sg,sg,sg, subgs, gs, nsg, box);


	p1 = tetra->v1;
	p2 = tetra->v2;
	p3 = tetra->v3;
	p4 = tetra->v4;

	double minx = min(min(min(p1.x, p2.x), p3.x), p4.x);
	double maxx = max(max(max(p1.x, p2.x), p3.x), p4.x);
	double miny = min(min(min(p1.y, p2.y), p3.y), p4.y);
	double maxy = max(max(max(p1.y, p2.y), p3.y), p4.y);
	double minz = min(min(min(p1.z, p2.z), p3.z), p4.z);
	double maxz = max(max(max(p1.z, p2.z), p3.z), p4.z);

	p1.x = minx;
	p1.y = miny;
	p1.z = minz;

	p2.x = minx;
	p2.y = miny;
	p2.z = maxz;

	p3.x = minx;
	p3.y = maxy;
	p3.z = minz;

	p4.x = minx;
	p4.y = maxy;
	p4.z = maxz;

	p5.x = maxx;
	p5.y = miny;
	p5.z = minz;

	p6.x = maxx;
	p6.y = miny;
	p6.z = maxz;

	p7.x = maxx;
	p7.y = maxy;
	p7.z = minz;

	p8.x = maxx;
	p8.y = maxy;
	p8.z = maxz;


	if(isInBox(p1, v1, v8, 2*dx2)
	|| isInBox(p2, v1, v8, 2*dx2)
	|| isInBox(p3, v1, v8, 2*dx2)
	|| isInBox(p4, v1, v8, 2*dx2)
	|| isInBox(p5, v1, v8, 2*dx2)
	|| isInBox(p6, v1, v8, 2*dx2)
	|| isInBox(p7, v1, v8, 2*dx2)
	|| isInBox(p8, v1, v8, 2*dx2)){
		return true;
	}

	return false;
}

 */
