#ifndef __RENDER__
#define __RENDER__
#include "types.h"
#include "isoplane.h"

class Render{
public:
    Render(int imagesize, REAL boxsize, TetraIsoPlane * isoplane,
           int * argc_, char * args_[]);
    ~Render();
    
    //show the plane on screen and also returns the plane
    float * showPlane(REAL isoval);
    
    //just return the redering result
    float * getPlane(REAL isoval);

    //return the running time for rendering
    double getRenderTime();

private:
    TetraIsoPlane * isoplane_;
    int imagesize_;
    REAL boxsize_;
    float * image_;
    float * colorImg_;
    double rendertime_;

};

#endif
