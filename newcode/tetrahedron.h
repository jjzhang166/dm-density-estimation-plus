/*
 * tetrahedron.h
 * This file defines the Tetrahedron structure.
 *
 *  Created on: Dec 17, 2012
 *      Author: lyang
 */

#ifndef TETRAHEDRON_H_
#define TETRAHEDRON_H_
#include <cstdlib>
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
    CUDA_CALLABLE_MEMBER REAL dot(const Point &other) const;
    CUDA_CALLABLE_MEMBER const Point cross(const Point &other) const;
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
	CUDA_CALLABLE_MEMBER bool isInTetra(const Point &p);
	CUDA_CALLABLE_MEMBER bool isInTetra(const Point &p, double &d0, double &d1, double &d2, double &d3, double &d4);


	CUDA_CALLABLE_MEMBER REAL minx();
	CUDA_CALLABLE_MEMBER REAL miny();
	CUDA_CALLABLE_MEMBER REAL minz();
	CUDA_CALLABLE_MEMBER REAL maxx();
	CUDA_CALLABLE_MEMBER REAL maxy();
	CUDA_CALLABLE_MEMBER REAL maxz();

	CUDA_CALLABLE_MEMBER Tetrahedron & operator=(const Tetrahedron & rhs);
	CUDA_CALLABLE_MEMBER Tetrahedron(const Tetrahedron &tetra);
    CUDA_CALLABLE_MEMBER static double getVolume(const Point &v1, const Point &v2, const Point &v3, const Point &v4);

private:
	REAL minx_, maxx_, miny_, maxy_, minz_, maxz_;

	//double d0;				// compute d0 first to reduce calculation
};



// Ind tetrahedron
class IndTetrahedron{
public:
    CUDA_CALLABLE_MEMBER IndTetrahedron(){
        ind1 = 0;
        ind2 = 0;
        ind3 = 0;
        ind4 = 0;
    };
    int ind1;
    int ind2;
    int ind3;
    int ind4;
    
	CUDA_CALLABLE_MEMBER IndTetrahedron & operator=(const IndTetrahedron & rhs);
	CUDA_CALLABLE_MEMBER IndTetrahedron(const IndTetrahedron &tetra);
};


// Ind Tetrahedron manager
// manage the index tetrahedron
class IndTetrahedronManager{
public:
    //isVelocity: does velocity participated in the calculation?
    CUDA_CALLABLE_MEMBER IndTetrahedronManager(Point * parray = NULL, REAL box = 3200, bool isVelocity = false, Point * varray = NULL);
    
    CUDA_CALLABLE_MEMBER void setIsVelocity(bool isVelocity);
    CUDA_CALLABLE_MEMBER void setPosArray(Point * parray);
    CUDA_CALLABLE_MEMBER void setVelArray(Point * varray);
    CUDA_CALLABLE_MEMBER void setBoxSize(REAL box);
    
    
    //Deprecate
    CUDA_CALLABLE_MEMBER void setRedShitDistortion(Point distortAxis);
    
    //test wheter the point is in this tetra
	CUDA_CALLABLE_MEMBER bool isInTetra(const IndTetrahedron &tetra_, const Point &p) const;
	CUDA_CALLABLE_MEMBER bool isInTetra(const IndTetrahedron &tetra_,
                                        Point &p, double &d0, double &d1,
                                        double &d2, double &d3, double &d4) const;
    
    CUDA_CALLABLE_MEMBER Point& posa(const IndTetrahedron &tetra_) const;
    CUDA_CALLABLE_MEMBER Point& posb(const IndTetrahedron &tetra_) const;
    CUDA_CALLABLE_MEMBER Point& posc(const IndTetrahedron &tetra_) const;
    CUDA_CALLABLE_MEMBER Point& posd(const IndTetrahedron &tetra_) const;
    
    CUDA_CALLABLE_MEMBER Point& vela(const IndTetrahedron &tetra_) const;
    CUDA_CALLABLE_MEMBER Point& velb(const IndTetrahedron &tetra_) const;
    CUDA_CALLABLE_MEMBER Point& velc(const IndTetrahedron &tetra_) const;
    CUDA_CALLABLE_MEMBER Point& veld(const IndTetrahedron &tetra_) const;
    
	CUDA_CALLABLE_MEMBER REAL minx(const IndTetrahedron &tetra_) const;
	CUDA_CALLABLE_MEMBER REAL miny(const IndTetrahedron &tetra_) const;
	CUDA_CALLABLE_MEMBER REAL minz(const IndTetrahedron &tetra_) const;
	CUDA_CALLABLE_MEMBER REAL maxx(const IndTetrahedron &tetra_) const;
	CUDA_CALLABLE_MEMBER REAL maxy(const IndTetrahedron &tetra_) const;
	CUDA_CALLABLE_MEMBER REAL maxz(const IndTetrahedron &tetra_) const;
    
    CUDA_CALLABLE_MEMBER REAL getVolume(const IndTetrahedron &tetra_) const;
    
    //compute and return how many periodical tetrahedrons
    CUDA_CALLABLE_MEMBER int getNumPeriodical(const IndTetrahedron &tetra_);
    //return the period tetrahedrons
	CUDA_CALLABLE_MEMBER Tetrahedron * getPeroidTetras(const IndTetrahedron &tetra_);
    
private:
    Tetrahedron tetras_p[8];    //used for solve the peoridical condition
    REAL box_;
    Point * positionArray;
    Point * velocityArray;
    bool isVelocity_;           //does velocity participated in the calculation?
    
};



#endif /* TETRAHEDRON_H_ */
