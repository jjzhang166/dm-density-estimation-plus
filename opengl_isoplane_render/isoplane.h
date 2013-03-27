#ifndef __ISOPLANE__
#define __ISOPLANE__
#include "types.h"
#include "tetrahedron.h"
#include "indtetrastream.h"
#include "triangle.h"

class TetraIsoPlane{
public:
    // memgridsize is the same 
    TetraIsoPlane(IndTetraStream * tetraStream);
    ~TetraIsoPlane();
    Triangle * getCurrentIsoPlane();
    Triangle * getIsoPlane(int i);  // get the i-th block isoplane
    int getTriangleNumbers();        // return the triangle numbers in current isoplane
    int getTotalBlockNum();          // has more tetrahedrons to cut?
    void setIsoValue(REAL isovalue);
    void loadIsoplane(int i);        // load the i-th block isovalue
    
private:
    Triangle * isoplane_;
    IndTetraStream * tetraStream_;
    int isoPlane_Size_;
    int currentIsoPlane_Size_;
    REAL isovalue_;
    int convertTetras2IsoPlane(REAL isovalue, 
                    Triangle *, 
                    IndTetrahedron *,
                    IndTetrahedronManager & tetramanager,
                    int nums); //returns the number of tetrahedrons in the isoplane
};
#endif
