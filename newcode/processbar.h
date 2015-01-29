/*
 * processbar.h
 *
 *  Created on: Dec 27, 2012
 *      Author: lyang
 */

#ifndef PROCESS_BAR_H_
#define PROCESS_BAR_H_
#include <iostream>

class ProcessBar {
public:
	ProcessBar(double maxvalue, int type);
	void start();
	void setvalue(double value);
	void end();
private:
	double maxvalue_;
	double currentvalue_;
	int currentpercent_;
	int type_;
};

#endif /* PROCESS_BAR_H_ */

