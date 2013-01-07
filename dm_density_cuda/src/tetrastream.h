/*
 * tetrastream.h
 *
 * This file defines TetraStream class, which takes a datafile as the input, load
 * as many data to the memory as the form of Tetrahedron. Output a Tetrahedron or
 * a point to the array of Tetrahedrons in current memory.
 *  Created on: Dec 17, 2012
 *      Author: lyang
 */

#ifndef TETRASTREAM_H_
#define TETRASTREAM_H_
#include <string>
#include <vector>
#include "tetrahedron.h"
#include "readgadget.h"
using namespace std;

/**
 * Get the tetrahedrons in a stream
 * The basic idea is reading the large number of particle datas in blocks
 */
class TetraStream {
public:
	TetraStream(std::string filename, int memgridsize);
	int getTotalBlockNum();		//get how many subblocks are there in total
	int getBlockSize();			//get the particle grid size in memory
	int getBlockNumTetra();		//get number of tetrahedrons in memory
	gadget_header getHeader();	//get the header
	Tetrahedron * getCurrentBlock();	//return the current tetrahedron block
	Tetrahedron * getBlock(int i);		//return the i-th tetrahedron block
	int getCurrentInd();				//return the current block id
	void loadBlock(int i);				//load the i-th block
	bool reset();						//return to the 0-th block
	~TetraStream();

private:
	std::string filename_;
	int mem_grid_size_;		// the grid_size are there in the memory
	int total_parts_;			// total particle numbers
	int particle_grid_size_;	// the total grid_size
	int mem_tetra_size_;		// the total tetras in the memory
	int total_tetra_grid_num_;	// how many tetra grids are there
	int current_tetra_num;		// how many available tetras are there in the current block

	Point * position_;								// the position datas
	//std::vector<Tetrahedron> tetras_;				// the tetrahedrons
	Tetrahedron * tetras_;
	//std::vector<Tetrahedron>::iterator tetra_iter_; // the iterator for tetra

	void convertToTetrahedron(int ii, int jj, int kk);	// convert the vertex data to tetrahedron
														// ii, jj, kk is max-min+1

	int current_ind_tetra;			// the current tetra index
	int current_ind_block;			// the current block index
	void addTetra(int ind1, int ind2, int ind3, int ind4);// add a tetra to the vector
	void addTetra(int i1, int j1, int k1, int i2, int j2, int k2, int i3,
			int j3, int k3, int i4, int j4, int k4,
			int isize, int jsize, int ksize);// add a tetra to the vector
	GSnap * gsnap_;

};

#endif /* TETRASTREAM_H_ */
