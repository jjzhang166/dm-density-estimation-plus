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

class Point{
public:
	double x;
	double y;
	double z;
};

class Tetrahedron{
public:
	Tetrahedron();
	double computeVolume();
	Point v1;
	Point v2;
	Point v3;
	Point v4;

	//test wheter the point is in this tetra
	bool isInTetra(Point p);
	double volume;

	double det4d(double m[4][4]);
	void c2m(Point p1, Point p2, Point p3, Point p4, double m[4][4]);

	double minx();
	double miny();
	double minz();
	double maxx();
	double maxy();
	double maxz();

private:
};


#endif /* TETRAHEDRON_H_ */
