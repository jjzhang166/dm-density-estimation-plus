/*TriConverter:
 Author: Lin Yang
 
 Convert a tetrahedron to triangles.
 Process a list of Tetrahedrons, and store the triangles into the memory.
 
 Copyright reserved.
 */


#ifndef __TRICONVERTER__
#define __TRICONVERTER__
#include <vector>
#include <iostream>
#include <fstream>
#include <string>
#include "types.h"
#include "tetracut.h"
#include "tetrahedron.h"
#include "triangle.h"

using namespace std;
//render the tetrahedrons into triangle file
//those triangles are 2D-triangle with velocity values
class TriConverter{
public:
    
    TriConverter( int imagesize,
                 float boxsize,
                 int maxNumTriangles = 1024*1024,
                 bool isVelocity = false
                 );
    
    ~TriConverter();
    
    //void    setOutput           (int ouputCode); //connected by "|"
    
    void process(Tetrahedron & tetra);
    void process(Tetrahedron * tetras, int numTetras);
    
    vector<int> & getTrianglePlaneIds();      //return a vector of the triangle ids
    
    vector<float> & getVertex();              //get a float array of the vertexes
    vector<float> & getDensity();             //get a float vector of densities
    vector<float> & getVelocityX();
    vector<float> & getVelocityY();
    vector<float> & getVelocityZ();
    
    int * getNumTrisInPlanes();               //get a array of number of triangles in each plane
    bool isReachMax();                        //whether the numoftris reach maximum
    void reset();                             //clear memories
    int getTotalTriangles();

private:

    
    vector<int> trianglePlaneIds_;
    vector<float> vertexData_;
    vector<float> densityData_;
    
    vector<float> velXData_;
    vector<float> velYData_;
    vector<float> velZData_;
    
    int * numTrianglePlanes;
    int maxNumTriangles_;
    int currentTriNum_;
    int totalTriangles_ ;
    
    int     imagesize_;
    int     numplanes_;
    REAL    boxsize_;
    REAL    dz_;
    REAL    startz_;
    
    IsoZCutter cutter;
    
    bool isVelocity_;
};

#endif
