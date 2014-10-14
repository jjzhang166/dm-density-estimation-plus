/*****************************************************************
 * tetracut.h: the definition of class IsoCutter, a class to cut 
 * the tetrahedron to triangles.
 * 
 * The general class is the first IsoCutter.
 * For specific use in the code, use IsoZCutter.
 *
 * Author: Lin Yang
 * Date: Feb 2014
 * Changes:
 * Oct/14/2014: 
 *   Add the a slab cut, to calculate the portion that this tetra-
 *   hedra occupies the slab region. This value is used to calc 
 *   the actual density in the final cube cell and avoid overflow
 *   if any the tetraheron is too small.
 *****************************************************************/


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
    
    //set the value of each point of the tetrahedra
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


//Only cut the tetrahedron along the z-direction
//This cutter will run very fast if the 4-point of the tetrahedron is already sorted along z-direction
class IsoZCutter{
public:
    IsoZCutter();
    void setTetrahedron(Tetrahedron *tetra);
    
    //sort on the tetrehadra's vertex based on the z-coordinates
    //if already sorted, this will return immediately
    void sortVertex();
    
    //return how many triangles are there after this cutting
    int cut(REAL isoz);
    
    //get the i-th triangles after the cut
    Triangle3d& getTriangle(int i);


    //cut the triangle with a thickness of the slice
    //this thickness is used to calucate the portion to occupy the slab
    //the portion is calcuted as (Volume intersecting the slab) /(A*t)
    //where A is the area of the triangle. If t->0, then the value is 1.
    //This class does not calculate the fraction but do return the height
    //of the triangle instead. The fraction will be calculated in the use
    //case of this cutter.
    void setThickness(REAL thickness); 
    REAL getTriangleThickness(int i);

private:
    REAL slabThickness_;
    Tetrahedron *tetra_;
    Triangle3d triangles_[2];
    REAL triangleThickness_[2];
    Point v12, v13, v14, v23, v24, v34;
    int num_tris_;
    REAL val[4];
};
#endif
