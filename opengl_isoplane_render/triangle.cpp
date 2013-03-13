#include "triangle.h"

CUDA_CALLABLE_MEMBER Triangle& Triangle::operator=(const Triangle &rhs){
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