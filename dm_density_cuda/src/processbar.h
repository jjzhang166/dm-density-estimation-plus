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
	ProcessBar(int maxvalue);
	void start();
	void setvalue(int value);
	void end();
private:
	int maxvalue_;
	double currentvalue_;
	int currentpercent_;
};

#endif /* PROCESS_BAR_H_ */