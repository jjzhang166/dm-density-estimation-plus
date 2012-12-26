/*
 * estimator.h
 * Estimater takes TetraStream, GridManager as input, compute a full density estimation of the data
 *  Created on: Dec 17, 2012
 *      Author: lyang
 */

#ifndef ESTIMATOR_H_
#define ESTIMATOR_H_

class Estimater{
public:
	Estimater(TetraStream * tetrastream, GridManager * gridmanager);
	void computeDensity();
	bool isGood();						// if false, has some error
	bool isFinished();					// whether the calculation is finished
private:
	TetraStream * tetrastream_;
	GridManager * gridmanager_;
	bool good_;
	bool finished_;
};


#endif /* ESTIMATOR_H_ */
