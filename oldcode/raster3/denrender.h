#ifndef __DENRENDER__
#define __DENRENDER__
#include <vector>
#include "types.h"
//#include "buffers.h"
#include "tetracut.h"
#include "tetrahedron.h"

//render the density through a serier plane cutter of the tetrahedrons
//startz, dz, and numplane specifies how many slices of the density field
//will be rendered

enum RenderType {
            DENSITY,                    //render density
            STREAM,                     //render stream
            VELOCITY_X,                 //render velocity x
            VELOCITY_Y,                 //render velocity y
            VELOCITY_Z                  //render velocity z
};

class DenRender{
public:
    
            DenRender            (
                                 int imagesize,
                                 float boxsize,
                                 float startz,
                                 float dz,
                                 int numplane,
                                 vector<RenderType> rentypes
                                 );
    
            ~DenRender          ();
    
    void    rend                (Tetrahedron & tetra);
    void    finish              ();
    
    float*  getResult           ();
    
    
    static const int VERTEXBUFFERDEPTH;
    
    static const int NUM_OF_RENDERTRYPE_LIMIT;
                                            //the limit of render
                                            //types of this render

private:
    
    void    rendplane(int i);
    
    void    init();
    
    int     imagesize_;
    REAL    boxsize_;
    int     numplanes_;
    
    //store the result
    float*  result_;
    
    //image stores only one slides of the field
    float   startz_;
    float   dz_;
    
    vector<RenderType> rentypes_;
    
    IsoZCutter cutter;
};

#endif
