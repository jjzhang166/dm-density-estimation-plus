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
using namespace std;

class Point{
public:
	REAL x;
	REAL y;
	REAL z;
	CUDA_CALLABLE_MEMBER Point & operator=(const Point &rhs);
	CUDA_CALLABLE_MEMBER Point(const Point &point);
	CUDA_CALLABLE_MEMBER Point();
    CUDA_CALLABLE_MEMBER const Point operator+(const Point &other) const;
    CUDA_CALLABLE_MEMBER const Point operator-(const Point &other) const;
    CUDA_CALLABLE_MEMBER const Point operator*(const REAL &other) const;
    CUDA_CALLABLE_MEMBER const Point operator/(const REAL &other) const;
    CUDA_CALLABLE_MEMBER const REAL dot(const Point &other) const;
};

class Tetrahedron{
public:
	CUDA_CALLABLE_MEMBER Tetrahedron();
	
	CUDA_CALLABLE_MEMBER REAL computeVolume();				//also compute minmax
	CUDA_CALLABLE_MEMBER void computeMaxMin();
	Point v1;
	Point v2;
	Point v3;
	Point v4;

	Point velocity1;
	Point velocity2;
	Point velocity3;
	Point velocity4;

		
	REAL volume;
	REAL invVolume;							// the inverse of the volume

	//test wheter the point is in this tetra
	CUDA_CALLABLE_MEMBER bool isInTetra(Point &p);
	CUDA_CALLABLE_MEMBER bool isInTetra(Point &p, double &d0, double &d1, double &d2, double &d3, double &d4);


	CUDA_CALLABLE_MEMBER REAL minx();
	CUDA_CALLABLE_MEMBER REAL miny();
	CUDA_CALLABLE_MEMBER REAL minz();
	CUDA_CALLABLE_MEMBER REAL maxx();
	CUDA_CALLABLE_MEMBER REAL maxy();
	CUDA_CALLABLE_MEMBER REAL maxz();

	CUDA_CALLABLE_MEMBER Tetrahedron & operator=(const Tetrahedron & rhs);
	CUDA_CALLABLE_MEMBER Tetrahedron(const Tetrahedron &tetra);

private:
	REAL minx_, maxx_, miny_, maxy_, minz_, maxz_;
	CUDA_CALLABLE_MEMBER double getVolume(Point &v1, Point &v2, Point &v3, Point &v4);

	//double d0;				// compute d0 first to reduce calculation
};


#endif /* TETRAHEDRON_H_ */
