/*
 * indtetrastream.h
 *
 * This file defines IndTetraStream class, which is a low memory 
 * conterpart of TetraStream. The tetrahedrons used in this stream
 * is the IndTetraHedron, so that the output will only takes small
 * amount of memory. And this should be faster then tetrastream.
 *
 *  Created on: Dec 17, 2012
 *      Author: lyang
 */

#ifndef TETRASTREAM_H_
#define TETRASTREAM_H_

#if defined(_WIN32) || defined(_WIN64)
#include "gettimeofday_win.h"
#else
#include <unistd.h>
#include <sys/time.h>
#endif


#include <string>
#include <vector>
#include "tetrahedron.h"
#include "readgadget.h"
using namespace std;

/**
 * Get the tetrahedrons in a stream
 * The basic idea is reading the large number of particle datas in blocks
 */
class IndTetraStream {
public:

    /********************************************************************
     * USE "isAllData" to avoid data COPYING. But need memory to store all
     * the data into memory.
     * WARNING: isAllData can only work in high memory mode
     ********************************************************************/
    //parttype - which type of the gadgetfile will be in use?
    //gridsize = -1: use npart^(1/3) as gridsize. Otherwise use this
	IndTetraStream(std::string filename,
                   int memgridsize,
                   int parttype = 1,
                   int gridsize = -1,
                   bool isVelocity = false,
                   bool isHighMem = true,
                   bool isAllData = true);
    
    
    
	int getTotalBlockNum();		//get how many subblocks are there in total
	int getBlockSize();			//get the particle grid size in memory
	int getBlockNumTetra();		//get number of tetrahedrons in memory
   
    Point * getPositionBlock();
    Point * getVelocityBlock();
    
	gadget_header getHeader();          //get the header

	IndTetrahedron * getCurrentBlock();	//return the current tetrahedron block
	IndTetrahedron * getBlock(int i);	//return the i-th tetrahedron block
	int getCurrentInd();				//return the current block id
    
                                        //get the current tetrahedron manager
                                        //which include the velocity block,
                                        //and position block
    IndTetrahedronManager& getCurrentIndTetraManager();
    
	void loadBlock(int i);				//load the i-th block
	bool reset();						//return to the 0-th block
    
    // currently takes no action here
	void setCorrection();
    
	void setIsInOrder(bool isinorder);		//set up whether the data is in order?
	~IndTetraStream();
    
    double getRunningTime(){
        return iotime_;
    };

private:
	std::string filename_;
    
    //current starting index
    int imin, jmin, kmin, imax, jmax, kmax;


	int mem_grid_size_;         // the grid_size are there in the memory
	int total_parts_;			// total particle numbers
	int particle_grid_size_;	// the total grid_size
	int mem_tetra_size_;		// the total tetras in the memory
	int total_tetra_grid_num_;	// how many tetra grids are there
	int current_tetra_num;		// how many available tetras are there in the current block
    

	Point * position_;								// the position datas
	Point * velocity_;								// velocities
    
	IndTetrahedron * tetras_;
    IndTetrahedronManager indTetraManager_;
    
    void convertToTetrahedron(int ii, int jj, int kk);	// convert the vertex data to tetrahedron
														// ii, jj, kk is max-min+1

	//void singleVoxvolCorrection();

	int current_ind_tetra;			// the current tetra index
	int current_ind_block;			// the current block index
	void addTetra(int ind1, int ind2, int ind3, int ind4);// add a tetra to the vector, do some corrections
	void addTetra(int i1, int j1, int k1, int i2, int j2, int k2, int i3,
			int j3, int k3, int i4, int j4, int k4,
			int isize, int jsize, int ksize);// add a tetra to the vector
	GSnap * gsnap_;

    //utility for add all the tetrahedron in a single vox
    void addTetraAllVox(int i, int j, int k, int ii, int jj, int kk);

	//single vox_vol correction
	//GridManager * grids_;
	REAL box;

	bool isPeriodical_;
	bool isInOrder_;
	bool isVelocity_;
    bool isAllData_;
    
    double iotime_, t0_, t1_;
    timeval timediff;
};


//this class has a lot of funtional similarity with TetraStream, but this one takes
//only small memeory
class TetraStreamer{
public:
    TetraStreamer(std::string filename,
                  int memgridsize,
                  int parttype = 1,
                  int gridsize = -1,
                  bool isHighMem = true,
                  bool isAllData = true,
                  bool isVelocity = true,
                  bool isCorrection = true,
                  bool isInOrder = false,
                  int limit_tetracount = 500000);
    ~TetraStreamer();
    
    bool hasNext();
    Tetrahedron * getNext(int& num_tetras_);
    
    IndTetraStream * getIndTetraStream(){
        return indstream_;
    };
    
    int getTetraContLimit(){
        return limit_tetracount_;
    };
    
    void reset();

private:
    IndTetraStream * indstream_;
    
    int current_block_id_;
    int current_tetra_id_;  //current indtetra_id in current block
    int total_block_num_;
    int total_tetra_num_;   //total indtetra_num in current block
    
    Tetrahedron * tetras_;  //tetrahedrons for returning the blocks
    IndTetrahedron * indtetras_;
    int tetra_count_;       //the tetrahedron count in the array
    int limit_tetracount_;
};
#endif /* TETRASTREAM_H_ */
