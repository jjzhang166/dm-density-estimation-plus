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

#define EPSILON 1e-6
#define EPSILON1 1e-11

CUDA_CALLABLE_MEMBER Tetrahedron::Tetrahedron(){
	volume = 0;
	minx_ = 0;
	miny_ = 0;
	minz_ = 0;
	maxx_ = 0;
	maxy_ = 0;
	maxz_ = 0;
}

CUDA_CALLABLE_MEMBER void Tetrahedron::computeMaxMin(){
	minx_ = min(min(min(v1.x, v2.x), v3.x), v4.x);
	miny_ = min(min(min(v1.y, v2.y), v3.y), v4.y);
	minz_ = min(min(min(v1.z, v2.z), v3.z), v4.z);
	maxx_ = max(max(max(v1.x, v2.x), v3.x), v4.x);
	maxy_ = max(max(max(v1.y, v2.y), v3.y), v4.y);
	maxz_ = max(max(max(v1.z, v2.z), v3.z), v4.z);
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
	vol /= 6.0;
	volume = abs(vol);

	//compute min and max
	computeMaxMin();
	
	//compute d0 to reduce calculation
	//double m[4][4];
	//c2m(v1, v2, v3, v4, m);		//change the det to be det / 10^11
	//d0 = det4d(m);

	return vol;
}

CUDA_CALLABLE_MEMBER double Tetrahedron::det4d(double m[4][4]) {
   double value;
   double v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12;
		 v1 =  (m[0][3] * m[1][2] * m[2][1] * m[3][0]-m[0][2] * m[1][3] * m[2][1] * m[3][0]);
		 v2 =  (-m[0][3] * m[1][1] * m[2][2] * m[3][0]+m[0][1] * m[1][3] * m[2][2] * m[3][0]);
		 v3 =  (+m[0][2] * m[1][1] * m[2][3] * m[3][0]-m[0][1] * m[1][2] * m[2][3] * m[3][0]);
		 v4 =  (-m[0][3] * m[1][2] * m[2][0] * m[3][1]+m[0][2] * m[1][3] * m[2][0] * m[3][1]);
		 v5 =  (+m[0][3] * m[1][0] * m[2][2] * m[3][1]-m[0][0] * m[1][3] * m[2][2] * m[3][1]);
		 v6 =  (-m[0][2] * m[1][0] * m[2][3] * m[3][1]+m[0][0] * m[1][2] * m[2][3] * m[3][1]);
		 v7 =  (+m[0][3] * m[1][1] * m[2][0] * m[3][2]-m[0][1] * m[1][3] * m[2][0] * m[3][2]);
		 v8 =  (-m[0][3] * m[1][0] * m[2][1] * m[3][2]+m[0][0] * m[1][3] * m[2][1] * m[3][2]);
		 v9 =  (+m[0][1] * m[1][0] * m[2][3] * m[3][2]-m[0][0] * m[1][1] * m[2][3] * m[3][2]);
		 v10 = (-m[0][2] * m[1][1] * m[2][0] * m[3][3]+m[0][1] * m[1][2] * m[2][0] * m[3][3]);
		 v11 = (+m[0][2] * m[1][0] * m[2][1] * m[3][3]-m[0][0] * m[1][2] * m[2][1] * m[3][3]);
		 v12 = (-m[0][1] * m[1][0] * m[2][2] * m[3][3]+m[0][0] * m[1][1] * m[2][2] * m[3][3]);
   value = (v1 + v2 + v3 + v4 +  v5 + v6 + v7 + v8 + v9 + v10 + v11 + v12);
   return value;
}

CUDA_CALLABLE_MEMBER void Tetrahedron::c2m(Point p1, Point p2, Point p3, Point p4, double m[4][4]){
	m[0][0] = p1.x * 1.0e-11;
	m[0][1] = p1.y * 1.0e-11;
	m[0][2] = p1.z * 1.0e-11;
	m[0][3] = 1.0 * 1.0e-11;
	m[1][0] = p2.x;
	m[1][1] = p2.y;
	m[1][2] = p2.z;
	m[1][3] = 1.0f;
	m[2][0] = p3.x;
	m[2][1] = p3.y;
	m[2][2] = p3.z;
	m[2][3] = 1.0f;
	m[3][0] = p4.x;
	m[3][1] = p4.y;
	m[3][2] = p4.z;
	m[3][3] = 1.0f;
}

CUDA_CALLABLE_MEMBER bool Tetrahedron::isInTetra(Point p){
	if(p.x > maxx() || p.y > maxy() || p.z > maxz()
	|| p.x < minx() || p.y < miny() || p.z < minz()){
		return false;
	}
	double m[4][4];
	double d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0;

	c2m(v1, v2, v3, v4, m);		//change the det to be det / 10^11
	d0 = det4d(m);

	c2m(p, v2, v3, v4, m);
	d1 = det4d(m);

	c2m(v1, p, v3, v4, m);
	d2 = det4d(m);

	c2m(v1, v2, p, v4, m);
	d3 = det4d(m);

	c2m(v1, v2, v3, p, m);
	d4 = det4d(m);


	if(d0 > 0){
		return (d1 >= 0) && (d2 >= 0) && (d3 >= 0) && (d4 >= 0);
	}else{
		return (d1 <= 0) && (d2 <= 0) && (d3 <= 0) && (d4 <= 0);
	}
}


CUDA_CALLABLE_MEMBER REAL Tetrahedron::minx(){
	return minx_;
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::miny(){
	return miny_;
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::minz(){
	return minz_;
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::maxx(){
	return maxx_;
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::maxy(){
	return maxy_;
}
CUDA_CALLABLE_MEMBER REAL Tetrahedron::maxz(){
	return maxz_;
}


CUDA_CALLABLE_MEMBER Point &  Point::operator=(const Point &rhs){
	this->x = rhs.x;
	this->y = rhs.y;
	this->z = rhs.z;
	return *this;
}

CUDA_CALLABLE_MEMBER Point::Point(const Point &point){
	this->x = point.x;
	this->y = point.y;
	this->z = point.z;
}


CUDA_CALLABLE_MEMBER Tetrahedron & Tetrahedron::operator=(const Tetrahedron & rhs){
	this->v1 = rhs.v1;
	this->v2 = rhs.v2;
	this->v3 = rhs.v3;
	this->v4 = rhs.v4;
	this->volume = rhs.volume;
	this->maxx_ = rhs.maxx_;
	this->maxy_ = rhs.maxy_;
	this->maxz_ = rhs.maxz_;
	this->minx_ = rhs.minx_;
	this->miny_ = rhs.miny_;
	this->minz_ = rhs.minz_;
	return *this;
}

CUDA_CALLABLE_MEMBER Tetrahedron::Tetrahedron(const Tetrahedron & rhs){
	this->v1 = rhs.v1;
	this->v2 = rhs.v2;
	this->v3 = rhs.v3;
	this->v4 = rhs.v4;
	this->volume = rhs.volume;
	this->maxx_ = rhs.maxx_;
	this->maxy_ = rhs.maxy_;
	this->maxz_ = rhs.maxz_;
	this->minx_ = rhs.minx_;
	this->miny_ = rhs.miny_;
	this->minz_ = rhs.minz_;
}

CUDA_CALLABLE_MEMBER Point::Point(){
	x = 0;
	y = 0;
	z = 0;
}