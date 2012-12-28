/*
 * tetrahedron.cpp
 *
 *  Created on: Dec 20, 2012
 *      Author: lyang
 */
#include <cmath>
#include <algorithm>
#include <cstdio>

using namespace std;

#include "tetrahedron.h"

Tetrahedron::Tetrahedron() {
	volume = 0;
}

double Tetrahedron::computeVolume() {
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

	vol = v1x * v2y * v3z + v1y * v2z * v3x + v1z * v2x * v3y
			- (v1z * v2y * v3x + v1y * v2x * v3z + v1x * v2z * v3y);
	//vol =  v1[0]*v2[1]*v3[2] + v1[1]*v2[2]*v3[0] + v1[2]*v2[0]*v3[1] - $
	//  (v1[2]*v2[1]*v3[0] + v1[1]*v2[0]*v3[2] + v1[0]*v2[2]*v3[1])

	vol /= 6.0;
	volume = abs(vol);
	return vol;
}

double Tetrahedron::det4d(double m[4][4]) {
	double value;
	value = m[0][3] * m[1][2] * m[2][1] * m[3][0]
			- m[0][2] * m[1][3] * m[2][1] * m[3][0]
			- m[0][3] * m[1][1] * m[2][2] * m[3][0]
			+ m[0][1] * m[1][3] * m[2][2] * m[3][0]
			+ m[0][2] * m[1][1] * m[2][3] * m[3][0]
			- m[0][1] * m[1][2] * m[2][3] * m[3][0]
			- m[0][3] * m[1][2] * m[2][0] * m[3][1]
			+ m[0][2] * m[1][3] * m[2][0] * m[3][1]
			+ m[0][3] * m[1][0] * m[2][2] * m[3][1]
			- m[0][0] * m[1][3] * m[2][2] * m[3][1]
			- m[0][2] * m[1][0] * m[2][3] * m[3][1]
			+ m[0][0] * m[1][2] * m[2][3] * m[3][1]
			+ m[0][3] * m[1][1] * m[2][0] * m[3][2]
			- m[0][1] * m[1][3] * m[2][0] * m[3][2]
			- m[0][3] * m[1][0] * m[2][1] * m[3][2]
			+ m[0][0] * m[1][3] * m[2][1] * m[3][2]
			+ m[0][1] * m[1][0] * m[2][3] * m[3][2]
			- m[0][0] * m[1][1] * m[2][3] * m[3][2]
			- m[0][2] * m[1][1] * m[2][0] * m[3][3]
			+ m[0][1] * m[1][2] * m[2][0] * m[3][3]
			+ m[0][2] * m[1][0] * m[2][1] * m[3][3]
			- m[0][0] * m[1][2] * m[2][1] * m[3][3]
			- m[0][1] * m[1][0] * m[2][2] * m[3][3]
			+ m[0][0] * m[1][1] * m[2][2] * m[3][3];
	return value;
}

void Tetrahedron::c2m(Point p1, Point p2, Point p3, Point p4, double m[4][4]) {
	m[0][0] = p1.x;
	m[0][1] = p1.y;
	m[0][2] = p1.z;
	m[0][3] = 1;
	m[1][0] = p2.x;
	m[1][1] = p2.y;
	m[1][2] = p2.z;
	m[1][3] = 1;
	m[2][0] = p3.x;
	m[2][1] = p3.y;
	m[2][2] = p3.z;
	m[2][3] = 1;
	m[3][0] = p4.x;
	m[3][1] = p4.y;
	m[3][2] = p4.z;
	m[3][3] = 1;
}

bool Tetrahedron::isInTetra(Point p) {
	double m[4][4];
	double d0, d1, d2, d3, d4;

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

	if (d0 > 0) {
		return (d1 >= 0) && (d2 >= 0) && (d3 >= 0) && (d4 >= 0);
	} else if(d0 < 0){
		return (d1 <= 0) && (d2 <= 0) && (d3 <= 0) && (d4 <= 0);
	}
	return false;
}

double Tetrahedron::minx() {
	return min(min(min(v1.x, v2.x), v3.x), v4.x);
}
double Tetrahedron::miny() {
	return min(min(min(v1.y, v2.y), v3.y), v4.y);
}
double Tetrahedron::minz() {
	return min(min(min(v1.z, v2.z), v3.z), v4.z);
}
double Tetrahedron::maxx() {
	return max(max(max(v1.x, v2.x), v3.x), v4.x);
}
double Tetrahedron::maxy() {
	return max(max(max(v1.y, v2.y), v3.y), v4.y);
}
double Tetrahedron::maxz() {
	return max(max(max(v1.z, v2.z), v3.z), v4.z);
}

