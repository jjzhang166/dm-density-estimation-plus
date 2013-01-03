/*
 * tetrahedron.h
 * This file defines the Tetrahedron structure.
 *
 *  Created on: Dec 17, 2012
 *      Author: lyang
 */

#ifndef TETRAHEDRON_H_
#define TETRAHEDRON_H_
#include <cmath>
#include "types.h"

class Point {
public:
	REAL x;
	REAL y;
	REAL z;
};

class Tetrahedron {
public:
	Tetrahedron();
	REAL computeVolume();
	Point v1;
	Point v2;
	Point v3;
	Point v4;

	//test wheter the point is in this tetra
	bool isInTetra(Point p);
	REAL volume;

	REAL det4d(REAL m[4][4]);
	void c2m(Point p1, Point p2, Point p3, Point p4, REAL m[4][4]);
	Point getCenter();

	REAL minx();
	REAL miny();
	REAL minz();
	REAL maxx();
	REAL maxy();
	REAL maxz();

private:
};

#endif /* TETRAHEDRON_H_ */
