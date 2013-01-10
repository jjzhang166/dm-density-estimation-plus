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
	Estimater(TetraStream * tetrastream, GridManager * gridmanager, int tetra_list_mem_lim);
	void computeDensity();
	bool isGood();						// if false, has some error
	bool isFinished();					// whether the calculation is finished
	void getRunnintTime(double &iotime, double &calctime);
private:
	TetraStream * tetrastream_;
	GridManager * gridmanager_;
	bool good_;
	bool finished_;
	double iotime_;
	double calctime_;
	int gpu_tetra_list_mem_lim;
};


#endif /* ESTIMATOR_H_ */
