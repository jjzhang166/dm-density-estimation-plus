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

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class Point{
public:
	REAL x;
	REAL y;
	REAL z;
	CUDA_CALLABLE_MEMBER Point & operator=(const Point &rhs);
	CUDA_CALLABLE_MEMBER Point(const Point &point);
	CUDA_CALLABLE_MEMBER Point();
};

class Tetrahedron{
public:
	CUDA_CALLABLE_MEMBER Tetrahedron();
	REAL computeVolume();
	Point v1;
	Point v2;
	Point v3;
	Point v4;

	//test wheter the point is in this tetra
	CUDA_CALLABLE_MEMBER bool isInTetra(Point p);
	REAL volume;

	CUDA_CALLABLE_MEMBER double det4d(double m[4][4]);
	CUDA_CALLABLE_MEMBER void c2m(Point p1, Point p2, Point p3, Point p4, double m[4][4]);

	CUDA_CALLABLE_MEMBER REAL minx();
	CUDA_CALLABLE_MEMBER REAL miny();
	CUDA_CALLABLE_MEMBER REAL minz();
	CUDA_CALLABLE_MEMBER REAL maxx();
	CUDA_CALLABLE_MEMBER REAL maxy();
	CUDA_CALLABLE_MEMBER REAL maxz();

	CUDA_CALLABLE_MEMBER Tetrahedron & operator=(const Tetrahedron & rhs);
	CUDA_CALLABLE_MEMBER Tetrahedron(const Tetrahedron &tetra);

private:
};


#endif /* TETRAHEDRON_H_ */
