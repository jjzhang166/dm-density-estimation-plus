#ifndef _LY_DTETRA__
#define _LY_DTETRA__
#include "io_utils.h"
#include "tetrahedron.h"


//the reduced H0, for the redshift distortion
//RH0 = (1 Mpc) / (1 code length units) / H0
#ifndef RH0
#define RH0 10
#endif




class DtetraStream{
public:
    DtetraStream(std::string basename);
    ~DtetraStream();
    void loadBlock(int i);  //load the i-th block
    float * getPosArray();
    float * getVelArray();
    divide_header getHeader();
    IndTetrahedronManager & getCurrentIndtetraManeger();
    
    //get number of intetrahedrons in current block
    int getNumTetras();
    
    //get the i-th indetrahedron
    void getIndTetra(IndTetrahedron & indtetra, int i);
    
    void setRedShitDistortion(Point distortAxis);
    
private:
    divide_header header_;
    Point * temppos_;
    Point * tempvel_;
    Point * pos_;
    Point * vel_;
    int64_t * inds_;
    IndTetrahedronManager manager_;
    std::string basename_;
    
    bool isReadShiftDistorted_;
    Point distortAxis_;
};

#endif