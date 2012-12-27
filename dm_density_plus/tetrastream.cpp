/*
 * tetrastream.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: lyang
 */
#include <string>
#include <vector>

using namespace std;

#include "tetrahedron.h"
#include "tetrastream.h"
//#include "gadget/gadgetreader.hpp"
#include "readgadget.h"

TetraStream::TetraStream(string filename, int inputmemgridsize){
	filename_ = filename;
	mem_grid_size_ = inputmemgridsize;
	mem_tetra_size_ = 6 * (mem_grid_size_) * (mem_grid_size_) * (mem_grid_size_);

	//TODO read from file
	total_parts_ = 0;
	particle_grid_size_ = 0;

	tetra_iter_ = tetras_.begin();
	numParts_ = 0;
	position_ = NULL;

	ngrid_ = 0;
	printf(">>> Read particle positions...\n");
	readPosition();			// read the index-sorted position data
	printf(">>> Converting particle to tetrahedrons... \n");
	convertToTetrahedron();	// convert the vertex data to tetrahedron
	//test
	//printf("---test---Tetrahedron numbers %d \n", tetras_.size());
	//printf("---test---Tetrahedron numbers %d \n", tetras_.size());
	//unsigned int i = 0;
	//for(i = 0; i < tetras_.size(); i++){
	//	printf("---test---Tetrahedron vol %f \n", tetras_[i].volume);
	//}
}

bool TetraStream::hasnext(){
	if(tetra_iter_ != tetras_.end()){
		return true;
	}else{
		return false;
	}
}

vector<Tetrahedron>::iterator TetraStream::next(){
	tetra_iter_ ++;
	return tetra_iter_;
}

bool TetraStream::reset(){
	tetra_iter_ = tetras_.begin();
	return true;
}

TetraStream::~TetraStream(){
	delete position_;
	delete gsnap_;
}

void TetraStream::readPosition(){
	//GadgetReader::GSnap gsnap(filename_);
	gsnap_ = new GSnap(filename_);
	int nparts = gsnap_->Npart;//gsnap.GetNpart(gsnap.GetFormat());

	float * data_array = gsnap_->pos;//new float[nparts * 3];
	uint32_t * ids = gsnap_->ids;//new int[nparts];

	//gsnap.GetBlock("POS ", data_array, nparts, 0, 0);
	//gsnap.GetBlock("ID  ", ids, nparts, 0, 0);

	ngrid_ = ceil(pow(nparts, 1.0/3.0));

	position_ = new Point[nparts];
	numParts_ = nparts;

	//sorting
	for(int i = 0; i < nparts; i++){
		position_[ids[i]].x = data_array[i * 3];
		position_[ids[i]].y = data_array[i * 3 + 1];
		position_[ids[i]].z = data_array[i * 3 + 2];
	}

	//delete data_array;


}

void TetraStream::addTetra(int ind1, int ind2, int ind3, int ind4){
	Tetrahedron tetra_;
	tetra_.v1 = position_[ind1];
	tetra_.v2 = position_[ind2];
	tetra_.v3 = position_[ind3];
	tetra_.v4 = position_[ind4];
	tetra_.computeVolume();
	tetras_.push_back(tetra_);
}


void TetraStream::addTetra(int i1, int j1, int k1,
		int i2, int j2, int k2,
		int i3, int j3, int k3,
		int i4, int j4, int k4){	// add a tetra to the vectot
	int ind1, ind2, ind3, ind4;
	ind1 = (i1) + (j1) * ngrid_ + (k1) * ngrid_ * ngrid_;
	ind2 = (i2) + (j2) * ngrid_ + (k2) * ngrid_ * ngrid_;
	ind3 = (i3) + (j3) * ngrid_ + (k3) * ngrid_ * ngrid_;
	ind4 = (i4) + (j4) * ngrid_ + (k4) * ngrid_ * ngrid_;
	addTetra(ind1, ind2, ind3, ind4);
}


void TetraStream::convertToTetrahedron(){


	int i, j, k;		//loop variables
	//printf("----test----Ngrid %d \n", ngrid_);

	for(k = 0; k < ngrid_ - 1; k++){
		for(j = 0; j < ngrid_ - 1; j++){
			for(i = 0; i < ngrid_ - 1; i++){
				//1
				addTetra(i,   j,   k,
						 i,   j+1, k,
						 i,   j,   k+1,
						 i+1, j,   k+1);
				//2
				addTetra(i,   j,   k,
						 i,   j+1, k,
						 i+1, j,   k+1,
						 i+1, j,   k);
				//3
				addTetra(i,   j,   k+1,
						 i,   j+1, k+1,
						 i+1, j,   k+1,
						 i,   j+1, k);
				//4
				addTetra(i,   j+1, k,
						 i+1, j,   k+1,
						 i+1, j+1, k+1,
						 i,   j+1, k+1);
				//5
				addTetra(i,   j+1, k,
						 i+1, j,   k+1,
						 i+1, j+1, k+1,
						 i+1, j+1, k);
				//6
				addTetra(i,   j+1, k,
						 i+1, j,   k+1,
						 i+1, j+1, k,
						 i+1, j,   k);

			}
		}
	}
}
