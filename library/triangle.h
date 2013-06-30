#ifndef __TRIANGLE__
#define __TRIANGLE__
#include "tetrahedron.h"

class Point2d{
public:
    REAL x;
    REAL y;
    
    CUDA_CALLABLE_MEMBER Point2d & operator=(const Point2d &rhs){
        x = rhs.x;
        y = rhs.y;
        return *this;
    };
    
    CUDA_CALLABLE_MEMBER Point2d & operator=(const Point &rhs){
        x = rhs.x;
        y = rhs.y;
        return *this;
    };
    
	CUDA_CALLABLE_MEMBER Point2d(const Point2d &point){
        x = point.x;
        y = point.y;
    };
	CUDA_CALLABLE_MEMBER Point2d(){
        x = 0;
        y = 0;
    };

    CUDA_CALLABLE_MEMBER const Point getPoint() const{
       Point p;
       p.x = x;
       p.y = y;
       p.z = 0;
       return p;
    };

};


class Triangle{
public:
    Point2d a;
    Point val1;
    Point2d b;
    Point val2;
    Point2d c;
    Point val3;
    CUDA_CALLABLE_MEMBER Triangle & operator=(const Triangle &rhs);
	CUDA_CALLABLE_MEMBER Triangle(const Triangle &tri);
	CUDA_CALLABLE_MEMBER Triangle();
    CUDA_CALLABLE_MEMBER REAL getArea();
};

//triangles in 3D
class Triangle3d{
public:
    Point a;
    Point val1;
    Point b;
    Point val2;
    Point c;
    Point val3;
    CUDA_CALLABLE_MEMBER Triangle3d & operator=(const Triangle3d &rhs);
	CUDA_CALLABLE_MEMBER Triangle3d(const Triangle3d &tri);
	CUDA_CALLABLE_MEMBER Triangle3d();

    CUDA_CALLABLE_MEMBER REAL getArea();
};


#endif
