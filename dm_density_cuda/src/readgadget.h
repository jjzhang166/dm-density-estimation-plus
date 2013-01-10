/*
 * readgadget.h
 *
 *  Created on: Dec 27, 2012
 *      Author: lyang
 */

#ifndef READGADGET_H_
#define READGADGET_H_
#include <fstream>
#include "types.h"
#include "tetrahedron.h"
#include "gadgetheader.h"

class GSnap {
public:
	GSnap(std::string filename);
	~GSnap();
	gadget_header header;
	uint32_t Npart;

	void readPosBlock(Point * &posblock, int imin, int jmin, int kmin, int imax, int jmax, int kmax, bool isPeriodical = true, bool isOrdered = false);

private:
	uint32_t * ids;
	string filename_;
	int grid_size;

	Point readPos(std::fstream &file, int count);

	void readIndex(std::fstream &file, int *block,
			int imin, int jmin, int kmin, int imax, int jmax, int kmax, bool isPeriodical = true, bool isOrdered = false);
};

#endif /* READGADGET_H_ */
