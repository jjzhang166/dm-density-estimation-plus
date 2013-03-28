#ifndef __ISOPLANE__
#define __ISOPLANE__
#include "types.h"
#include "tetrahedron.h"
#include "indtetrastream.h"
#include "triangle.h"

class TetraIsoPlane{
public:
    // memgridsize is the same 
    TetraIsoPlane(IndTetraStream * tetraStream, int isoplane_mem_size = 1000000);
    ~TetraIsoPlane();

    // each time, use loadIsoPlane, the output will be stored in a stream. 
    // use hasNext, and getNextIsoPlaneBlock to get all isoplane block
    void loadIsoplane(int i);        // load the i-th block isovalue
    bool hasNext();
    Triangle * getNextIsoPlaneBlock(int & num_triangles);
    //Triangle * getIsoPlane(int i);  // get the i-th block isoplane

    int getTotalBlockNum();          // has more tetrahedrons to cut?
    int getTriangleNumbers();        // return the triangle numbers in current isoplane
    void setIsoValue(REAL isovalue);
    double getCutTime(){
        return cuttingtime_;
    };
    
    IndTetraStream * getIndStream(){
        return tetraStream_;
    };
    
private:
    double cuttingtime_;

    Triangle * isoplane_;
    IndTetraStream * tetraStream_;
    int isoPlane_Size_;
    int currentIsoPlane_Size_;
    REAL isovalue_;

    int isoplane_mem_size_;
    
    void convertTetras2IsoPlane(); //returns the number of tetrahedrons in the isoplane
    
    //these variables will be reset when calling loadIsoplane
    int total_tetra_num_;
    int current_tetra_num_;
    //IndTetrahedronManager * tetramanager_;
    IndTetrahedron * tetras;
    
};
#endif
