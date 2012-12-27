/*
 * readgadget.h
 *
 *  Created on: Dec 27, 2012
 *      Author: lyang
 */

#ifndef READGADGET_H_
#define READGADGET_H_
#include "gadgetheader.h"

class GSnap{
public:
	GSnap(std::string filename);
	~GSnap();
	gadget_header header;
	uint64_t * ids;
	float * pos;
	float * vel;
	uint32_t Npart;
};


#endif /* READGADGET_H_ */
