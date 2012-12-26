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
using namespace std;

class TetraStream{
public:
	TetraStream(std::string filename, int memgridsize);
	bool hasnext();
	std::vector<Tetrahedron>::iterator next();
	std::vector<Tetrahedron> * getTretras();
	bool reset();
	~TetraStream();

private:
	std::string filename_;
	long mem_grid_size_;		// the grid_size are there in the memory
	long total_parts_;			// total particle numbers
	int particle_grid_size_;	// the total grid_size
	int mem_tetra_size_;		// the total tetras in the memory

	Point * position_;								// the position datas
	int numParts_;									//number of particles
	std::vector<Tetrahedron> tetras_;				// the tetrahedrons
	std::vector<Tetrahedron>::iterator tetra_iter_; // the iterator for tetra
	int ngrid_;										// particle gridsize

	void readPosition();		// read the index-sorted position data
	void convertToTetrahedron();// convert the vertex data to tetrahedron
	void addTetra(int ind1, int ind2, int ind3, int ind4);	// add a tetra to the vector
	void addTetra(int i1, int j1, int k1,
			int i2, int j2, int k2,
			int i3, int j3, int k3,
			int i4, int j4, int k4);	// add a tetra to the vector

};


#endif /* TETRASTREAM_H_ */
