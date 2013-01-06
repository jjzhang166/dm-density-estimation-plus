/*
 * tetrahedron.cpp
 *
 *  Created on: Dec 20, 2012
 *      Author: lyang
 */
#include <cmath>
#include <algorithm>
#include <cstdio>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#include "tetrahedron.h"

Tetrahedron::Tetrahedron(){
	volume = 0;
}

CUDA_CALLABLE_MEMBER REAL Tetrahedron::computeVolume(){
	REAL vol;
	REAL v1x, v1y, v1z;
	REAL v2x, v2y, v2z;
	REAL v3x, v3y, v3z;

	v1x = v2.x - v1.x;
	v1y = v2.y - v1.y;
	v1z = v2.z - v1.z;

	v2x = v3.x - v1.x;
	v2y = v3.y - v1.y;
	v2z = v3.z - v1.z;

	v3x = v4.x - v1.x;
	v3y = v4.y - v1.y;
	v3z = v4.z - v1.z;

	vol =  v1x*v2y*v3z + v1y*v2z*v3x + v1z*v2x*v3y -
	      (v1z*v2y*v3x + v1y*v2x*v3z + v1x*v2z*v3y);
	//vol =  v1[0]*v2[1]*v3[2] + v1[1]*v2[2]*v3[0] + v1[2]*v2[0]*v3[1] - $
    //  (v1[2]*v2[1]*v3[0] + v1[1]*v2[0]*v3[2] + v1[0]*v2[2]*v3[1])

	vol /= 6.0;
	volume = abs(vol);
	return vol;
}

CUDA_CALLABLE_MEMBER REAL Tetrahedron::det4d(REAL m[4][4]) {
   REAL value;
   value =
		  m[0][3] * m[1][2] * m[2][1] * m[3][0]-m[0][2] * m[1][3] * m[2][1] * m[3][0]
		 -m[0][3] * m[1][1] * m[2][2] * m[3][0]+m[0][1] * m[1][3] * m[2][2] * m[3][0]+
		  m[0][2] * m[1][1] * m[2][3] * m[3][0]-m[0][1] * m[1][2] * m[2][3] * m[3][0]-
		  m[0][3] * m[1][2] * m[2][0] * m[3][1]+m[0][2] * m[1][3] * m[2][0] * m[3][1]+
		  m[0][3] * m[1][0] * m[2][2] * m[3][1]-m[0][0] * m[1][3] * m[2][2] * m[3][1]-
		  m[0][2] * m[1][0] * m[2][3] * m[3][1]+m[0][0] * m[1][2] * m[2][3] * m[3][1]+
		  m[0][3] * m[1][1] * m[2][0] * m[3][2]-m[0][1] * m[1][3] * m[2][0] * m[3][2]-
		  m[0][3] * m[1][0] * m[2][1] * m[3][2]+m[0][0] * m[1][3] * m[2][1] * m[3][2]+
		  m[0][1] * m[1][0] * m[2][3] * m[3][2]-m[0][0] * m[1][1] * m[2][3] * m[3][2]-
		  m[0][2] * m[1][1] * m[2][0] * m[3][3]+m[0][1] * m[1][2] * m[2][0] * m[3][3]+
		  m[0][2] * m[1][0] * m[2][1] * m[3][3]-m[0][0] * m[1][2] * m[2][1] * m[3][3]-
		  m[0][1] * m[1][0] * m[2][2] * m[3][3]+m[0][0] * m[1][1] * m[2][2] * m[3][3];
   return value;
}

CUDA_CALLABLE_MEMBER void Tetrahedron::c2m(Point p1, Point p2, Point p3, Point p4, REAL m[4][4]){
	m[0][0] = p1.x;
	m[0][1] = p1.y;
	m[0][2] = p1.z;
	m[0][3] = 1.0;
	m[1][0] = p2.x;
	m[1][1] = p2.y;
	m[1][2] = p2.z;
	m[1][3] = 1.0;
	m[2][0] = p3.x;
	m[2][1] = p3.y;
	m[2][2] = p3.z;
	m[2][3] = 1.0;
	m[3][0] = p4.x;
	m[3][1] = p4.y;
	m[3][2] = p4.z;
	m[3][3] = 1.0;
}

CUDA_CALLABLE_MEMBER bool Tetrahedron::isInTetra(Point p){
	REAL m[4][4];
	REAL d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0;

	//[[3,2,1,1],[2,2,3,1],[5,4,3,1],[2,1,2,1]]
	//test
	//v1.x=3;v1.y=2;v1.z=1;
	//v2.x=2;v2.y=2;v3.z=3;
	//v3.x=5;v3.y=4;v3.z=3;
	//v4.x=2;v4.y=1;v4.z=2;

	c2m(v1, v2, v3, v4, m);
	d0 = det4d(m);

	//printf("Test Determinant %f / 4.00\n", d0);

	c2m(p, v2, v3, v4, m);
	d1 = det4d(m);

	c2m(v1, p, v3, v4, m);
	d2 = det4d(m);

	c2m(v1, v2, p, v4, m);
	d3 = det4d(m);

	c2m(v1, v2, v3, p, m);
	d4 = det4d(m);

	//testing
	//d2++;

	if(d0 > 0){
		return (d1 >= 0) && (d2 >= 0) && (d3 >= 0) && (d4 >= 0);
	}else{
		return (d1 <= 0) && (d2 <= 0) && (d3 <= 0) && (d4 <= 0);
	}
	return false;
}


CUDA_CALLABLE_MEMBER REAL Tetrahedron::minx(){
	return min(min(min(v1.x, v2.x), v3.x), v4.x);
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::miny(){
	return min(min(min(v1.y, v2.y), v3.y), v4.y);
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::minz(){
	return min(min(min(v1.z, v2.z), v3.z), v4.z);
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::maxx(){
	return max(max(max(v1.x, v2.x), v3.x), v4.x);
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::maxy(){
	return max(max(max(v1.y, v2.y), v3.y), v4.y);
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::maxz(){
	return max(max(max(v1.z, v2.z), v3.z), v4.z);
}

/*
CUDA_CALLABLE_MEMBER Point &  Point::operator=(const Point &rhs){
	this->x = rhs.x;
	this->y = rhs.y;
	this->z = rhs.z;
	return *this;
}



CUDA_CALLABLE_MEMBER Tetrahedron & Tetrahedron::operator=(const Tetrahedron & rhs){
	this->v1 = rhs.v1;
	this->v2 = rhs.v2;
	this->v3 = rhs.v3;
	this->v4 = rhs.v4;
	this->volume = rhs.volume;
	return *this;
}*/