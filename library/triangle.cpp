#include <cmath>
#include <cstdlib>
#include "triangle.h"

using namespace std;

CUDA_CALLABLE_MEMBER Triangle& Triangle::operator=(const Triangle &rhs){
	a = rhs.a;
	b = rhs.b;
	c = rhs.c;
    
    val1 = rhs.val1;
    val2 = rhs.val2;
    val3 = rhs.val3;
	return *this;
}

CUDA_CALLABLE_MEMBER Triangle& Triangle::operator=(const Triangle3d &rhs){
	a = rhs.a;
	b = rhs.b;
	c = rhs.c;
    
    val1 = rhs.val1;
    val2 = rhs.val2;
    val3 = rhs.val3;
	return *this;
}


CUDA_CALLABLE_MEMBER Triangle::Triangle(const Triangle &tri){
	a = tri.a;
	b = tri.b;
	c = tri.c;
    val1 = tri.val1;
    val2 = tri.val2;
    val3 = tri.val3;
}

CUDA_CALLABLE_MEMBER Triangle::Triangle(){
    Point zero;
    zero.x = 0;
    zero.y = 0;
    zero.z = 0;

    a = zero;
	b = zero;
	c = zero;
    val1 = zero;
    val2 = zero;
    val3 = zero;
}

CUDA_CALLABLE_MEMBER Triangle3d& Triangle3d::operator=(const Triangle3d &rhs){
	a = rhs.a;
	b = rhs.b;
	c = rhs.c;
    
    val1 = rhs.val1;
    val2 = rhs.val2;
    val3 = rhs.val3;
	return *this;
}

CUDA_CALLABLE_MEMBER Triangle3d::Triangle3d(const Triangle3d &tri){
	a = tri.a;
	b = tri.b;
	c = tri.c;
    val1 = tri.val1;
    val2 = tri.val2;
    val3 = tri.val3;
}

CUDA_CALLABLE_MEMBER Triangle3d::Triangle3d(){
    Point zero;
    zero.x = 0;
    zero.y = 0;
    zero.z = 0;

    a = zero;
	b = zero;
	c = zero;
    val1 = zero;
    val2 = zero;
    val3 = zero;
}

CUDA_CALLABLE_MEMBER REAL Triangle::getArea(){
    Point v1 = b.getPoint() - a.getPoint();
    Point v2 = c.getPoint() - a.getPoint();
    Point area = v1.cross(v2);
    return (REAL) (sqrt(area.dot(area)) / 2.0);
}

CUDA_CALLABLE_MEMBER REAL Triangle3d::getArea(){
    Point v1 = b - a;
    Point v2 = c - a;
    Point area = v1.cross(v2);
    return (REAL)(sqrt(area.dot(area)) / 2.0);
}


