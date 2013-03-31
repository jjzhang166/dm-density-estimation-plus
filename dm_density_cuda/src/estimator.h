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
	Estimater(TetraStreamer * tetrastream, GridManager * gridmanager);
	Estimater(TetraStreamer * tetrastream, GridManager * gridmanager, int tetra_list_mem_lim);
	Estimater(TetraStreamer * tetrastream, GridManager * gridmanager, GridVelocityManager * gridvelocity, int tetra_list_mem_lim);

	void computeDensity();
	bool isGood();						// if false, has some error
	bool isFinished();					// whether the calculation is finished
	void getRunnintTime(double &iotime, double &calctime);
	void setVerbose(bool verbose);
	void setIsVelocity(bool isvel);				// set whether calc
private:
	void initialize(TetraStreamer * tetrastream, GridManager * gridmanager);

	TetraStreamer * tetrastream_;
	GridManager * gridmanager_;
	GridVelocityManager * gridvelocity_;

	bool good_;
	bool finished_;
	double iotime_;
	double calctime_;
	int gpu_tetra_list_mem_lim;
	bool isVerbose_;
	bool isVelocity_;

};


#endif /* ESTIMATOR_H_ */
