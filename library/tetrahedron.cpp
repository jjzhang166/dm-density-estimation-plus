/*
 * tetrahedron.cpp
 *
 *  Created on: Dec 20, 2012
 *      Author: lyang
 */
#include <cmath>
#include <algorithm>
#include <cstdio>

//#include "cuda_runtime.h"
//#include "device_launch_parameters.h"

using namespace std;

#include "tetrahedron.h"
#include "tetrastream.h"

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

	if(volume != 0.0){
		invVolume = (REAL) 1.0 / volume;
	}else{
		invVolume = 0.0;
	}

	//compute min and max
	computeMaxMin();
	
	//compute d0 to reduce calculation
	//d0 = (REAL)getVolume(v1, v2, v3, v4);

	return vol;
}

CUDA_CALLABLE_MEMBER double Tetrahedron::getVolume(const Point &v1, const Point &v2, const Point &v3, const Point &v4){
	double vol;
	double v1x, v1y, v1z;
	double v2x, v2y, v2z;
	double v3x, v3y, v3z;

	v1x = v2.x - v1.x;
	v1y = v2.y - v1.y;
	v1z = v2.z - v1.z;

	v2x = v3.x - v1.x;
	v2y = v3.y - v1.y;
	v2z = v3.z - v1.z;

	v3x = v4.x - v1.x;
	v3y = v4.y - v1.y;
	v3z = v4.z - v1.z;

	vol = -(v1x*v2y*v3z + v1y*v2z*v3x + v1z*v2x*v3y -
	      (v1z*v2y*v3x + v1y*v2x*v3z + v1x*v2z*v3y));
	return vol;
}

CUDA_CALLABLE_MEMBER bool Tetrahedron::isInTetra(const Point &p, double &d0, double &d1, double &d2, double &d3, double &d4){
	if(p.x > maxx() || p.y > maxy() || p.z > maxz()
	|| p.x < minx() || p.y < miny() || p.z < minz()){
		return false;
	}

	//c2m(v1, v2, v3, v4, m);		//change the det to be det / 10^11
	d0 = getVolume(v1, v2, v3, v4);//det4d(m);

	//c2m(p, v2, v3, v4, m);
	d1 = getVolume(p, v2, v3, v4);

	//c2m(v1, p, v3, v4, m);
	d2 = getVolume(v1, p, v3, v4);

	//c2m(v1, v2, p, v4, m);
	d3 = getVolume(v1, v2, p, v4);

	//c2m(v1, v2, v3, p, m);
	d4 = getVolume(v1, v2, v3, p);

    double z0 = d0 * EPSILON;

	if(d0 > z0){
		return (d1 >= z0) && (d2 >= z0) && (d3 >= z0) && (d4 >= z0);
	}else{
		return (d1 <= -z0) && (d2 <= -z0) && (d3 <= -z0) && (d4 <= -z0);
	}
}

CUDA_CALLABLE_MEMBER bool Tetrahedron::isInTetra(const Point &p){
	//double m[4][4];
	double d0=0.0, d1=0.0, d2=0.0, d3=0.0, d4=0.0;
	return isInTetra(p, d0, d1, d2, d3, d4);
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
	x = rhs.x;
	y = rhs.y;
	z = rhs.z;
	return *this;
}

CUDA_CALLABLE_MEMBER Point::Point(const Point &point){
	x = point.x;
	y = point.y;
	z = point.z;
}



CUDA_CALLABLE_MEMBER Tetrahedron & Tetrahedron::operator=(const Tetrahedron & rhs){
	this->v1 = rhs.v1;
	this->v2 = rhs.v2;
	this->v3 = rhs.v3;
	this->v4 = rhs.v4;

	this->velocity1 = rhs.velocity1;
	this->velocity2 = rhs.velocity2;
	this->velocity3 = rhs.velocity3;
	this->velocity4 = rhs.velocity4;

	this->volume = rhs.volume;
	this->invVolume = rhs.invVolume;

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
	this->invVolume = rhs.invVolume;
    
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


CUDA_CALLABLE_MEMBER const Point Point::operator+(const Point &other) const{
    Point result = *this;
    result.x += other.x;
    result.y += other.y;
    result.z += other.z;
    return result;
}

CUDA_CALLABLE_MEMBER const Point Point::operator-(const Point &other) const{
    Point result = *this;
    result.x -= other.x;
    result.y -= other.y;
    result.z -= other.z;
    return result;
}

CUDA_CALLABLE_MEMBER const Point Point::operator*(const REAL &other) const{
    Point result = *this;
    result.x *= other;
    result.y *= other;
    result.z *= other;
    return result;
}

CUDA_CALLABLE_MEMBER const Point Point::operator/(const REAL &other) const{
    Point result = *this;
    result.x /= other;
    result.y /= other;
    result.z /= other;
    return result;
}


CUDA_CALLABLE_MEMBER REAL Point::dot(const Point &other) const{
    Point result = *this;
    result.x *= other.x;
    result.y *= other.y;
    result.z *= other.z;
    return result.x + result.y + result.z;
}

CUDA_CALLABLE_MEMBER const Point Point::cross(const Point &b) const{
    Point a = *this;
    Point res;
    res.x = a.y * b.z - a.z * b.y;
    res.y = a.z * b.x - a.x * b.z;
    res.z = a.x * b.y - a.y * b.x;

    return res;
}

