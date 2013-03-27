/*
 * IndTetraStream.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: lyang
 */
#include <string>
#include <vector>
#include <cmath>

using namespace std;

#include "tetrahedron.h"
#include "indtetrastream.h"
#include "readgadget.h"

//inputmemgridsize should be a divisor of the total_grid_size
IndTetraStream::IndTetraStream(string filename, int inputmemgridsize, bool isVelocity) {
	isVelocity_ = isVelocity;

	filename_ = filename;

	gsnap_ = new GSnap(filename_);
	particle_grid_size_ = (int)ceil(pow(gsnap_->Npart, 1.0 / 3.0));
	total_parts_ = gsnap_->Npart;
	current_tetra_num = 0;

	current_ind_tetra = 0;
	current_ind_block = 0;
    
    printf("Particle Data Grid Size %d\n", particle_grid_size_);

    mem_grid_size_ = inputmemgridsize;
	mem_tetra_size_ = 6 * (mem_grid_size_) * (mem_grid_size_)
        * (mem_grid_size_);

    
	tetras_ = new IndTetrahedron[mem_tetra_size_];
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
    
    indTetraManager_.setBoxSize(box);
    indTetraManager_.setIsVelocity(isVelocity_);
    indTetraManager_.setVelArray(velocity_);
    indTetraManager_.setPosArray(position_);
    
    //write tetrahedrons
    //tetrahedron grids are not changed during the work
	//convertToTetrahedron(mem_grid_size_ + 1,
    //                     mem_grid_size_ + 1,
    //                     mem_grid_size_ + 1);
    
}

void IndTetraStream::setIsInOrder(bool isinorder){
	isInOrder_ = isinorder;
}

bool IndTetraStream::reset() {
	loadBlock(0);
	return true;
}

IndTetraStream::~IndTetraStream() {
	delete position_;
	delete gsnap_;
	delete tetras_;
	if(isVelocity_){
		delete velocity_;
	}
}

int IndTetraStream::getTotalBlockNum(){
	return total_tetra_grid_num_;
}

int IndTetraStream::getBlockSize(){
	return mem_grid_size_;
}

int IndTetraStream::getBlockNumTetra(){
	return current_tetra_num;
}

IndTetrahedron * IndTetraStream::getCurrentBlock(){
	return tetras_;
}

IndTetrahedron * IndTetraStream::getBlock(int i){
	loadBlock(i);
	return tetras_;
}

int IndTetraStream::getCurrentInd(){
	return current_ind_block;
}

void IndTetraStream::loadBlock(int i){
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
    
    convertToTetrahedron(imax - imin + 1, jmax - jmin + 1, kmax - kmin + 1);
	current_ind_block = i;
}

void IndTetraStream::addTetra(int ind1, int ind2, int ind3, int ind4) {		
    
    tetras_[current_ind_tetra].ind1 = ind1;
    tetras_[current_ind_tetra].ind2 = ind2;
    tetras_[current_ind_tetra].ind3 = ind3;
    tetras_[current_ind_tetra].ind4 = ind4;
    current_ind_tetra ++;
}

void IndTetraStream::addTetra(int i1, int j1, int k1, int i2, int j2, int k2,
		int i3, int j3, int k3, int i4, int j4, int k4,
		int isize, int jsize, int ksize) {// add a tetra to the vectot
	int ind1, ind2, ind3, ind4;
	ind1 = (i1) + (j1) * isize + (k1) * isize * jsize;
	ind2 = (i2) + (j2) * isize + (k2) * isize * jsize;
	ind3 = (i3) + (j3) * isize + (k3) * isize * jsize;
	ind4 = (i4) + (j4) * isize + (k4) * isize * jsize;
	addTetra(ind1, ind2, ind3, ind4);
}

void IndTetraStream::convertToTetrahedron(int ii, int jj, int kk) {
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

void IndTetraStream::setCorrection(/*GridManager * grid*/){
    box = getHeader().BoxSize;
	isPeriodical_ = true;
}

gadget_header IndTetraStream::getHeader(){
	return this->gsnap_->header;	//get the header
}

IndTetrahedronManager& IndTetraStream::getCurrentIndTetraManager(){
    return indTetraManager_;
}

Point * IndTetraStream::getPositionBlock(){
    return position_;
}
Point * IndTetraStream::getVelocityBlock(){
    return velocity_;
}

