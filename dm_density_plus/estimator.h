/*
 * estimator.h
 * Estimater takes TetraStream, GridManager as input, compute a full density estimation of the data
 *  Created on: Dec 17, 2012
 *      Author: lyang
 */

#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_
#include "grids.h"
#include "tetraselector.h"

class Estimater {
public:
	Estimater(TetraStream * tetrastream, GridManager * gridmanager);
	void computeDensity();
	bool isGood();						// if false, has some error
	bool isFinished();					// whether the calculation is finished
	~Estimater();
private:
	TetraStream * tetrastream_;
	GridManager * gridmanager_;
	//TetraSelector * tetraselector_;

	bool good_;
	bool finished_;
	bool testTouch(Tetrahedron * tetra);
	double box;
	double ng;
	double dx2;

};

#endif /* ESTIMATOR_H_ */
