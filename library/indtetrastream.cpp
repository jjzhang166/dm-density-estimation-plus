/*
 * IndTetraStream.cpp
 *
 *  Created on: Dec 18, 2012
 *      Author: lyang
 */
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>

using namespace std;

#include "tetrahedron.h"
#include "indtetrastream.h"
#include "readgadget.h"


//the reduced H0, for the redshift distortion
//RH0 = (1 Mpc) / (1 code length units) / H0
#ifndef RH0
#define RH0 10
#endif

void IndTetraStream::init(){
    isRedshiftDistorted_ = false;
    isReshiftDistortionCalculated_ = false;
    
    //fprintf(stderr,"what's up\n");
	particle_grid_size_ = (int)ceil(pow(gsnap_->Npart, 1.0 / 3.0));
	total_parts_ = gsnap_->Npart;
	current_tetra_num = 0;
    
	current_ind_tetra = 0;
	current_ind_block = 0;
    
    fprintf(stderr,"Particle Data Grid Size %d\n", particle_grid_size_);
    
	if(mem_grid_size_ == -1){
		mem_grid_size_ = particle_grid_size_;
	}
    
	mem_tetra_size_ = 6 * (mem_grid_size_) * (mem_grid_size_)
    * (mem_grid_size_);
    
    fprintf(stderr,"Allocating %d memory size of indtetrahedron\n", mem_tetra_size_); 
	tetras_ = new IndTetrahedron[mem_tetra_size_];
	
    //if use all data, no need to build new pointers
    if(!isAllData_){
        position_ = new Point[(mem_grid_size_ + 1) * (mem_grid_size_ + 1) * (mem_grid_size_ + 1)];
        if(isVelocity_){
            velocity_ = new Point[(mem_grid_size_ + 1) * (mem_grid_size_ + 1) *     (mem_grid_size_ + 1)];
        }
        for(int i = 0; i < (mem_grid_size_ + 1) * (mem_grid_size_ + 1) * (mem_grid_size_ + 1); i++){
            position_[i].x = position_[i].y = position_[i].z = 0.0;
            if(isVelocity_){
                velocity_[i].x = velocity_[i].y = velocity_[i].z = 0.0;
            }
        }
    }
    
    
	total_tetra_grid_num_ = particle_grid_size_ / mem_grid_size_ *
    particle_grid_size_ / mem_grid_size_ *
    particle_grid_size_ / mem_grid_size_;
    
	//grids_ = NULL;
    
	isPeriodical_ = false;
	isInOrder_ = false;
    
    
    //write tetrahedrons
    //tetrahedron grids are not changed during the work
    //if use all the data, then tetrahedron index will change each load
    if(!isAllData_){
        convertToTetrahedron(mem_grid_size_ + 1,
                             mem_grid_size_ + 1,
                             mem_grid_size_ + 1);
    }else{
        position_ = gsnap_->getAllPos();
        velocity_ = gsnap_->getAllVel();
    }
   
    //for(int i = 0; i < 100; i ++ ){
    //    fprintf(stderr,"Position: %f %f %f\n", position_[i].x, position_[i].y, position_[i].z);
    //}

    indTetraManager_.setBoxSize(getHeader().BoxSize);
    indTetraManager_.setIsVelocity(isVelocity_);
    indTetraManager_.setVelArray(velocity_);
    indTetraManager_.setPosArray(position_);
}

IndTetraStream::IndTetraStream(std::string prefix,
               std::string basename,
               int numofFiles,
               int inputmemgridsize,
               int parttype,
               int gridsize,
               bool isVelocity,
               bool isHighMem,
               bool isAllData){
    
    
    isVelocity_ = isVelocity;
    if(isHighMem){
        isAllData_ = isAllData;
    }else{
        isAllData_ = false;
    }
    
    
    iotime_ = 0;
    
	gsnap_ = new GSnap(prefix, basename, numofFiles, isHighMem, parttype, gridsize);
    
    mem_grid_size_ = inputmemgridsize;
    particle_grid_size_ = gridsize;
    
    init();
    
}

//inputmemgridsize should be a divisor of the total_grid_size
//inputmemgridsize is the limit of data in memory
IndTetraStream::IndTetraStream(string filename,
                               int inputmemgridsize,
                               int parttype,
                               int gridsize,
                               bool isVelocity,
                               bool isHighMem,
                               bool isAllData) {
	isVelocity_ = isVelocity;
    if(isHighMem){
        isAllData_ = isAllData;
    }else{
        isAllData_ = false;
    }
    
    iotime_ = 0;
    filename_ = filename;
	gsnap_ = new GSnap(filename_, isHighMem, parttype, gridsize);
    
    mem_grid_size_ = inputmemgridsize;
    particle_grid_size_ = gridsize;
    
    init();
}


void IndTetraStream::getRedshiftDistoredPoint(Point & target,
                                              Point & velocity,
                                              Point & distortAxis,
                                              float redshift,
                                              float boxSize
                                              ){
    float a = 1.0 / (1.0 + redshift);
    
    Point displacement = distortAxis
    * velocity.dot(distortAxis)
    * sqrt(a) * RH0;// ; //to kpc/h
    
    target = target + displacement;
    
    target.x = fmod(target.x + boxSize, boxSize);
    target.y = fmod(target.y + boxSize, boxSize);
    target.z = fmod(target.z + boxSize, boxSize);
}


void IndTetraStream::setRedshiftDistort(Point distortAxis){
    isRedshiftDistorted_ = true;
    redshiftDistortAxis_ = distortAxis;
    if(isAllData_){
        if(!isReshiftDistortionCalculated_){
            for(int i = 0; i < total_parts_; i++){
                //test
                //fprintf(stdout,"Position_Ori: %f %f %f\n", position_[i].x, position_[i].y, position_[i].z);
                
                //fprintf(stdout,"Velocity: %f %f %f\n", velocity_[i].x, velocity_[i].y, velocity_[i].z);
                
                getRedshiftDistoredPoint(position_[i],
                                         velocity_[i],
                                         redshiftDistortAxis_,
                                         getHeader().redshift,
                                         getHeader().BoxSize);
                //test
                //fprintf(stdout,"Position_final: %f %f %f\n", position_[i].x, position_[i].y, position_[i].z);
            }
            isReshiftDistortionCalculated_ = true;
        }
    }
}

void IndTetraStream::setIsInOrder(bool isinorder){
	isInOrder_ = isinorder;
}

bool IndTetraStream::reset() {
	loadBlock(0);
	return true;
}

IndTetraStream::~IndTetraStream() {

	delete gsnap_;
	delete tetras_;
    
    if(!isAllData_){
        delete position_;
        if(isVelocity_){
            delete velocity_;
        }
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
    
    current_ind_block = i;
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

    gettimeofday(&timediff, NULL);
    t0_ = timediff.tv_sec + timediff.tv_usec / 1.0e6;
    
    //is use all data, no need load data each time.
    if(!isAllData_){
        if(!isVelocity_){
            gsnap_->readPosBlock(position_, imin, jmin, kmin, imax, jmax, kmax, isPeriodical_, isInOrder_);
        }else{
            gsnap_->readBlock(position_, velocity_, imin, jmin, kmin, imax, jmax, kmax, isPeriodical_, isInOrder_); 
            if(isRedshiftDistorted_){
                for(int l = 0; l < (mem_grid_size_ + 1) * (mem_grid_size_ + 1) * (mem_grid_size_ + 1); l++){
                    getRedshiftDistoredPoint(position_[l],
                                             velocity_[l],
                                             redshiftDistortAxis_,
                                             getHeader().redshift,
                                             getHeader().BoxSize);
                }
            }
        }
    }else
    {
        convertToTetrahedron(imax - imin + 1, jmax - jmin + 1, kmax - kmin + 1);
    }
    
    gettimeofday(&timediff, NULL);
    t1_ = timediff.tv_sec + timediff.tv_usec / 1.0e6;
    iotime_ += t1_ - t0_;
    
    //current_tetra_num = 6 * (imax - imin) * (jmax - imin) * (kmax - kmin);
    //fprintf(stderr,">>>%d %d\n", 6 * (imax - imin) * (jmax - jmin) * (kmax - kmin), 6*mem_grid_size_*mem_grid_size_*mem_grid_size_);



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
    if(!isAllData_){
        ind1 = (i1) + (j1) * isize + (k1) * isize * jsize;
        ind2 = (i2) + (j2) * isize + (k2) * isize * jsize;
        ind3 = (i3) + (j3) * isize + (k3) * isize * jsize;
        ind4 = (i4) + (j4) * isize + (k4) * isize * jsize;
    }else{
        ind1 = ((i1+imin) % particle_grid_size_) +
                ((j1+jmin)  % particle_grid_size_) * particle_grid_size_ +
                ((k1+kmin)  % particle_grid_size_) * particle_grid_size_ * particle_grid_size_;
        
        ind2 = ((i2+imin) % particle_grid_size_) +
                ((j2+jmin) % particle_grid_size_) * particle_grid_size_ +
                ((k2+kmin) % particle_grid_size_) * particle_grid_size_ * particle_grid_size_;
        
        ind3 = ((i3+imin) % particle_grid_size_) +
                ((j3+jmin) % particle_grid_size_) * particle_grid_size_ +
                ((k3+kmin) % particle_grid_size_) * particle_grid_size_ * particle_grid_size_;
        
        ind4 = ((i4+imin) % particle_grid_size_) +
                ((j4+jmin) % particle_grid_size_) * particle_grid_size_ +
                ((k4+kmin) % particle_grid_size_) * particle_grid_size_ * particle_grid_size_;
    }
	addTetra(ind1, ind2, ind3, ind4);
}

void IndTetraStream::addTetraAllVox(int i, int j, int k, int ii, int jj, int kk){
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


void IndTetraStream::convertToTetrahedron(int ii, int jj, int kk) {
	current_ind_tetra = 0;
	int i, j, k;		//loop variables

	for (k = 0; k < kk-1; k++) {
		for (j = 0; j < jj-1; j++) {
			for (i = 0; i < ii-1; i++) {
                addTetraAllVox(i, j, k, ii, jj, kk);
			}
		}
	}

	current_tetra_num = current_ind_tetra;
    //fprintf(stderr,"%d\n", current_tetra_num);

}

void IndTetraStream::setCorrection(/*GridManager * grid*/){
    box = getHeader().BoxSize;
	isPeriodical_ = true;
}

gadget_header IndTetraStream::getHeader(){
	return this->gsnap_->header;	//get the header
}

IndTetrahedronManager& IndTetraStream::getCurrentIndTetraManager(){
    indTetraManager_.setBoxSize(getHeader().BoxSize);
    indTetraManager_.setIsVelocity(isVelocity_);
    indTetraManager_.setVelArray(velocity_);
    indTetraManager_.setPosArray(position_);
    return indTetraManager_;
}

Point * IndTetraStream::getPositionBlock(){
    return position_;
}
Point * IndTetraStream::getVelocityBlock(){
    return velocity_;
}


TetraStreamer::~TetraStreamer(){
    delete indstream_;
    delete tetras_;
}



TetraStreamer::TetraStreamer(std::string prefix,
                             std::string basename,
                             int numOfFiles,
                             int memgridsize,
                             int parttype,
                             int gridsize,
                             bool isHighMem,
                             bool isAllData,
                             bool isVelocity,
                             bool isCorrection,
                             bool isInOrder,
                             int limit_tetracount){
    
    indstream_ = new IndTetraStream(prefix, basename, numOfFiles,
                                    memgridsize, parttype, gridsize,
                                    isVelocity, isHighMem, isAllData);
    
    if(isCorrection){
        indstream_->setCorrection();
    }
    indstream_->setIsInOrder(isInOrder);
    
    
    limit_tetracount_ = limit_tetracount;
    tetras_ = new Tetrahedron[limit_tetracount_];
    
    total_block_num_ = indstream_->getTotalBlockNum();
    current_block_id_ = -1;
    current_tetra_id_ = 0;
    total_tetra_num_ = 0;
    
}

TetraStreamer::TetraStreamer(std::string filename,
                             int memgridsize,
                             int parttype,
                             int gridsize,
                             bool isHighMem,
                             bool isAllData,
                             bool isVelocity,
                             bool isCorrection,
                             bool isInOrder,
                             int limit_tetracount){
    indstream_ = new IndTetraStream(filename, memgridsize, parttype, gridsize,isVelocity, isHighMem, isAllData);
    
    if(isCorrection){
        indstream_->setCorrection();
    }
    indstream_->setIsInOrder(isInOrder);
    
    
    limit_tetracount_ = limit_tetracount;
    tetras_ = new Tetrahedron[limit_tetracount_];
    
    total_block_num_ = indstream_->getTotalBlockNum();
    current_block_id_ = -1;
    current_tetra_id_ = 0;
    total_tetra_num_ = 0;
}


bool TetraStreamer::hasNext(){
    if(current_block_id_ < total_block_num_ - 1){
        return true;
    }else if(current_block_id_ == total_block_num_ - 1){
        if(current_tetra_id_ < total_tetra_num_){
            return true;
        }else{
            return false;
        }
    }else{
        return false;
    }
}

Tetrahedron* TetraStreamer::getNext(int& num_tetras_){
    
    IndTetrahedronManager * tetramanager = &(indstream_->
        getCurrentIndTetraManager());
    
    int count = 0;
    while((count < limit_tetracount_ - 8) &&
          (current_block_id_ < total_block_num_)){
    

        if(current_tetra_id_ >= total_tetra_num_){
            current_block_id_ ++;
            if(current_block_id_ < total_block_num_){
                current_tetra_id_ = 0;
                indstream_->loadBlock(current_block_id_);
                total_tetra_num_ = indstream_->getBlockNumTetra();
                indtetras_ = indstream_->getCurrentBlock();
                tetramanager = &(indstream_->getCurrentIndTetraManager());
            }else{
                current_tetra_id_ = 0;
                total_tetra_num_ = 0;
                break;
            }
        }
        
        //test
        //fprintf(stderr,"%d %d\n", current_tetra_id_, total_tetra_num_);
        
        if(tetramanager->posa(indtetras_[current_tetra_id_]).x < 0 ||
           tetramanager->posb(indtetras_[current_tetra_id_]).x < 0 ||
           tetramanager->posc(indtetras_[current_tetra_id_]).x < 0 ||
           tetramanager->posd(indtetras_[current_tetra_id_]).x < 0){
            //This tetrahedron is ignored
        }else{
            int temp_num_tetra = tetramanager->getNumPeriodical(indtetras_[current_tetra_id_]);
            
            //test
            //fprintf(stderr,"%d\n", temp_num_tetra);
            
            Tetrahedron * period_tetras = tetramanager->getPeroidTetras(indtetras_[current_tetra_id_]);
            
            //this may add 8 tetrahedrons
            for(int j = 0; j<temp_num_tetra; j++){
                tetras_[count] = period_tetras[j];
                count ++;
            }
        }
        
        current_tetra_id_ ++;
    }
    tetra_count_ = count;
    num_tetras_ = tetra_count_;
    return tetras_;
}

void TetraStreamer::reset(){
    total_block_num_ = indstream_->getTotalBlockNum();
    current_block_id_ = -1;
    current_tetra_id_ = 0;
    total_tetra_num_ = 0;
}

void TetraStreamer::setRedshiftDistort(Point distortAxis){
    indstream_->setRedshiftDistort(distortAxis);
}
