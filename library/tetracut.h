#ifndef __TETRACUT__H
#define __TETRACUT__H
#include "types.h"
#include "tetrahedron.h"
#include "triangle.h"
//this class is used to cut a tetraheron with a isovalue plan
//There will be 0, 1 or 2 triangles as output
class IsoCutter{
public:
    IsoCutter();
    void setTetrahedron(Tetrahedron *tetra);
    void setValues(const REAL v1, const REAL v2, const REAL v3, const REAL v4);
    
    //return how many triangles are there after this cutting 
    int cut(REAL isovalue);
    
    //cut the tetrahedron with value interpolated to the value of the triangle
    int cut(REAL isovalue, REAL val1, REAL val2, REAL val3, REAL val4);
    int cut(REAL isovalue, Point &val1, Point &val2, Point &val3, Point &val4);
    
    
    //get the i-th triangles after the cut
    Triangle3d& getTrangle(int i);

    bool testTriangle(const Point &a, const Point &b, const Point &c, const Point &d);
    
    //cut the line between p1, p2 with value v1, v2. Returns the point in retp
    //if no intersection, return false, else return true
    bool iso_cut_line(REAL _isoval, Point &p1, Point &p2, REAL v1, REAL v2, Point &retp);
    bool iso_cut_line(REAL _isoval, Point &p1, Point &p2, REAL v1, REAL v2, REAL val1, REAL val2, Point &retp, REAL &retv);
    bool iso_cut_line(REAL _isoval, Point &p1, Point &p2, REAL v1, REAL v2, Point &val1, Point &val2, Point &retp, Point &retv);
    
    
private:
    Tetrahedron *tetra_;
    REAL v1_, v2_, v3_, v4_;
    Triangle3d t1_, t2_;
    int num_tris_;

    
    bool iso_cut_line(REAL _isoval, Point &p1, Point &p2, REAL v1, REAL v2, Point &retp, REAL &t_par);
     
};
#endif
