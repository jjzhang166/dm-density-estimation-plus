/*
 * tetrastream.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: lyang
 */
#include <string>
#include <vector>
#include <cmath>

using namespace std;

#include "tetrahedron.h"
#include "tetrastream.h"
#include "readgadget.h"

//inputmemgridsize should be a divisor of the total_grid_size
TetraStream::TetraStream(string filename, int inputmemgridsize, bool isVelocity) {
	isVelocity_ = isVelocity;

	filename_ = filename;
	mem_grid_size_ = inputmemgridsize;
	mem_tetra_size_ = 6 * (mem_grid_size_) * (mem_grid_size_)
			* (mem_grid_size_) * 8; //8 is for the periodical condition

	gsnap_ = new GSnap(filename_);
	particle_grid_size_ = (int)ceil(pow(gsnap_->Npart, 1.0 / 3.0));
	total_parts_ = gsnap_->Npart;
	current_tetra_num = 0;

	current_ind_tetra = 0;
	current_ind_block = 0;

	tetras_ = new Tetrahedron[mem_tetra_size_];
	position_ = new Point[(mem_grid_size_ + 1) * (mem_grid_size_ + 1) * (mem_grid_size_ + 1)];
	if(isVelocity_){
		velocity_ = new Point[(mem_grid_size_ + 1) * (mem_grid_size_ + 1) * (mem_grid_size_ + 1)];
	}

	for(int i = 0; i < (mem_grid_size_ + 1) * (mem_grid_size_ + 1) * (mem_grid_size_ + 1); i++){
		position_[i].x = position_[i].y = position_[i].z = 0.0;
		if(isVelocity_){
			velocity_[i].x = velocity_[i].y = velocity_[i].z = 0.0;
		}
	}

	total_tetra_grid_num_ = particle_grid_size_ / mem_grid_size_ *
			particle_grid_size_ / mem_grid_size_ *
			particle_grid_size_ / mem_grid_size_;

	//grids_ = NULL;

	isPeriodical_ = false;
	isInOrder_ = false;

}

void TetraStream::setIsInOrder(bool isinorder){
	isInOrder_ = isinorder;
}

bool TetraStream::reset() {
	loadBlock(0);
	return true;
}

TetraStream::~TetraStream() {
	delete position_;
	delete gsnap_;
	delete tetras_;
	if(isVelocity_){
		delete velocity_;
	}
}

int TetraStream::getTotalBlockNum(){
	return total_tetra_grid_num_;
}

int TetraStream::getBlockSize(){
	return mem_grid_size_;
}

int TetraStream::getBlockNumTetra(){
	return current_tetra_num;
}

Tetrahedron * TetraStream::getCurrentBlock(){
	return tetras_;
}

Tetrahedron * TetraStream::getBlock(int i){
	loadBlock(i);
	return tetras_;
}

int TetraStream::getCurrentInd(){
	return current_ind_block;
}

void TetraStream::loadBlock(int i){
	if(i >= this->total_tetra_grid_num_){
		return;
	}
	int imin, jmin, kmin, imax, jmax, kmax;
	int ngb = particle_grid_size_ / mem_grid_size_;
	imin = i % ngb * mem_grid_size_;
	jmin = i / ngb % ngb * mem_grid_size_;
	kmin = i / ngb / ngb % ngb * mem_grid_size_;
	imax = imin + mem_grid_size_;

	//periodical condition
	if(imax == particle_grid_size_){
		//imax = particle_grid_size_ - 1;
	}
	jmax = jmin + mem_grid_size_;
	if(jmax == particle_grid_size_){
		//jmax = particle_grid_size_ - 1;
	}
	kmax = kmin + mem_grid_size_;
	if(kmax == particle_grid_size_){
		//kmax = particle_grid_size_ - 1;
	}

	if(!isVelocity_){
		gsnap_->readPosBlock(position_, imin, jmin, kmin, imax, jmax, kmax, isPeriodical_, isInOrder_);
	}else{
		gsnap_->readBlock(position_, velocity_, imin, jmin, kmin, imax, jmax, kmax, isPeriodical_, isInOrder_); 
	}

	//for(int ffi =0; ffi < ((imax - imin + 1)*(jmax - jmin + 1)*(kmax - kmin + 1)); ffi ++){
	//	printf("---%e\n", position_[ffi].x);
	//}


	convertToTetrahedron(imax - imin + 1, jmax - jmin + 1, kmax - kmin + 1);
	current_ind_block = i;
}

void TetraStream::splitTetraX(Tetrahedron & tetra, Tetrahedron & tetra1){
	static Point * vertexs[4];
	static Point * temp_;
	vertexs[0] = &(tetra.v1);
	vertexs[1] = &(tetra.v2);
	vertexs[2] = &(tetra.v3);
	vertexs[3] = &(tetra.v4);
	for(int i = 0; i < 4; i ++){
		for(int j = i+1; j < 4; j++){
			if((vertexs[i]->x) < (vertexs[j]->x)){
				temp_ = vertexs[j];
				vertexs[j] = vertexs[i];
				vertexs[i] = temp_;
			}
		}
	}
	vertexs[3]->x += box;
	if(vertexs[3]->x - vertexs[2]->x > box / 2.0){
		vertexs[2]->x += box;
		if(vertexs[2]->x - vertexs[1]->x > box / 2.0){
			vertexs[1]->x += box;
		}
	}
	tetra.computeVolume();

	tetra1 = tetra;
	tetra1.v1.x -= box;
	tetra1.v2.x -= box;
	tetra1.v3.x -= box;
	tetra1.v4.x -= box;
	tetra1.computeMaxMin();

}

void TetraStream::splitTetraY(Tetrahedron & tetra, Tetrahedron & tetra1){
	static Point * vertexs[4];
	static Point * temp_;
	vertexs[0] = &(tetra.v1);
	vertexs[1] = &(tetra.v2);
	vertexs[2] = &(tetra.v3);
	vertexs[3] = &(tetra.v4);
	for(int i = 0; i < 4; i ++){
		for(int j = i+1; j < 4; j++){
			if((vertexs[i]->y) < (vertexs[j]->y)){
				temp_ = vertexs[j];
				vertexs[j] = vertexs[i];
				vertexs[i] = temp_;
			}
		}
	}
	vertexs[3]->y += box;
	if(vertexs[3]->y - vertexs[2]->y > box / 2.0){
		vertexs[2]->y += box;
		if(vertexs[2]->y - vertexs[1]->y > box / 2.0){
			vertexs[1]->y += box;
		}
	}
	tetra.computeVolume();

	tetra1 = tetra;
	tetra1.v1.y -= box;
	tetra1.v2.y -= box;
	tetra1.v3.y -= box;
	tetra1.v4.y -= box;
	tetra1.computeMaxMin();

}

void TetraStream::splitTetraZ(Tetrahedron & tetra, Tetrahedron & tetra1){
	static Point * vertexs[4];
	static Point * temp_;
	vertexs[0] = &(tetra.v1);
	vertexs[1] = &(tetra.v2);
	vertexs[2] = &(tetra.v3);
	vertexs[3] = &(tetra.v4);
	for(int i = 0; i < 4; i ++){
		for(int j = i+1; j < 4; j++){
			if((vertexs[i]->z) < (vertexs[j]->z)){
				temp_ = vertexs[j];
				vertexs[j] = vertexs[i];
				vertexs[i] = temp_;
			}
		}
	}
	vertexs[3]->z += box;
	if(vertexs[3]->z - vertexs[2]->z > box / 2.0){
		vertexs[2]->z += box;
		if(vertexs[2]->z - vertexs[1]->z > box / 2.0){
			vertexs[1]->z += box;
		}
	}
	tetra.computeVolume();

	tetra1 = tetra;
	tetra1.v1.z -= box;
	tetra1.v2.z -= box;
	tetra1.v3.z -= box;
	tetra1.v4.z -= box;
	tetra1.computeMaxMin();

}

void TetraStream::addTetra(int ind1, int ind2, int ind3, int ind4) {
	static Tetrahedron tetras_p[8];	//for periodical correction
	Tetrahedron &tetra_ = tetras_p[0];
	tetra_.v1 = position_[ind1];
	tetra_.v2 = position_[ind2];
	tetra_.v3 = position_[ind3];
	tetra_.v4 = position_[ind4];

	if(isVelocity_){
		tetra_.velocity1 = velocity_[ind1];
		tetra_.velocity2 = velocity_[ind2];
		tetra_.velocity3 = velocity_[ind3];
		tetra_.velocity4 = velocity_[ind4];
	}
	tetra_.computeVolume();

	//add the tetrahedrons
	/*if(grids_ == NULL){
		tetras_[current_ind_tetra] = (tetra_);
		current_ind_tetra ++;
	}
	//correction
	else*/{
		//periodical correction:
		int tetra_num = 1;
		int temp_num = 0;
		if(tetra_.maxx() - tetra_.minx() > box / 2.0){
			splitTetraX(tetra_, tetras_p[1]);
			tetra_num ++;
		}
		temp_num = 0;
		for(int i = 0; i < tetra_num; i++){
			Tetrahedron &t = tetras_p[i];
			if(t.maxy() - t.miny() > box / 2.0){
				splitTetraY(t, tetras_p[tetra_num + temp_num]);
				temp_num ++;
			}
		}
		tetra_num += temp_num;
		temp_num = 0;
		for(int i = 0; i < tetra_num; i++){
			Tetrahedron &t = tetras_p[i];
			if(t.maxz() - t.minz() > box / 2.0){
				splitTetraZ(t, tetras_p[tetra_num + temp_num]);
				temp_num ++;
			}
		}
		tetra_num += temp_num;

        
        /*int outputGridSize = grids_->getGridSize();
        double outputBoxSize = grids_->getEndPoint().x - grids_->getStartPoint().x;*/
		for(int i = 0; i < tetra_num; i++){
			Tetrahedron &t = tetras_p[i];
			//single vox_vol
			/*int xindmin = (int)((t.minx() - grids_->getStartPoint().x) / outputBoxSize * outputGridSize);
			int xindmax = (int)((t.maxx() - grids_->getStartPoint().x) / outputBoxSize * outputGridSize);
			int yindmin = (int)((t.miny() - grids_->getStartPoint().y) / outputBoxSize * outputGridSize);
			int yindmax = (int)((t.maxy() - grids_->getStartPoint().y) / outputBoxSize * outputGridSize);
			int zindmin = (int)((t.minz() - grids_->getStartPoint().z) / outputBoxSize * outputGridSize);
			int zindmax = (int)((t.maxz() - grids_->getStartPoint().z) / outputBoxSize * outputGridSize);
			int n_samples = (xindmax - xindmin + 1) * (yindmax - yindmin + 1) * (zindmax - zindmin + 1);
			if(n_samples == 1){
				grids_->setValueByActualCoor(xindmin, yindmin, zindmin, 6.0f / vox_vol);
			}else*/{
				tetras_[current_ind_tetra] = t;
				current_ind_tetra ++;
				//printf("%d\n", current_ind_tetra);
			}
		}
		
	}
}

void TetraStream::addTetra(int i1, int j1, int k1, int i2, int j2, int k2,
		int i3, int j3, int k3, int i4, int j4, int k4,
		int isize, int jsize, int ksize) {// add a tetra to the vectot
	int ind1, ind2, ind3, ind4;
	ind1 = (i1) + (j1) * isize + (k1) * isize * jsize;
	ind2 = (i2) + (j2) * isize + (k2) * isize * jsize;
	ind3 = (i3) + (j3) * isize + (k3) * isize * jsize;
	ind4 = (i4) + (j4) * isize + (k4) * isize * jsize;
	addTetra(ind1, ind2, ind3, ind4);
}

void TetraStream::convertToTetrahedron(int ii, int jj, int kk) {
	current_ind_tetra = 0;
	int i, j, k;		//loop variables

	for (k = 0; k < kk-1; k++) {
		for (j = 0; j < jj-1; j++) {
			for (i = 0; i < ii-1; i++) {
				//1
				addTetra(i, j, k, i, j + 1, k, i, j, k + 1, i + 1, j, k + 1,
						ii, jj, kk);
				//2
				addTetra(i, j, k, i, j + 1, k, i + 1, j, k + 1, i + 1, j, k,
						ii, jj, kk);
				//3
				addTetra(i, j, k + 1, i, j + 1, k + 1, i + 1, j, k + 1, i,
						j + 1, k, ii, jj, kk);
				//4
				addTetra(i, j + 1, k, i + 1, j, k + 1, i + 1, j + 1, k + 1, i,
						j + 1, k + 1, ii, jj, kk);
				//5
				addTetra(i, j + 1, k, i + 1, j, k + 1, i + 1, j + 1, k + 1,
						i + 1, j + 1, k, ii, jj, kk);
				//6
				addTetra(i, j + 1, k, i + 1, j, k + 1, i + 1, j + 1, k, i + 1,
						j, k, ii, jj, kk);

			}
		}
	}
	current_tetra_num = current_ind_tetra;
}

void TetraStream::setCorrection(/*GridManager * grid*/){
	//this->grids_ = grid;
	//double boxOfGrids = (REAL)(grids_->getEndPoint().x - grids_->getStartPoint().x);
    box = getHeader().BoxSize;
	//ng = (REAL)grids_->getGridSize();
	//vox_vol = boxOfGrids * boxOfGrids * boxOfGrids / ng / ng / ng;
	isPeriodical_ = true;
}

gadget_header TetraStream::getHeader(){
	return this->gsnap_->header;	//get the header
}
