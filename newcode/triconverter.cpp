#include <cstdio>
#include <cstdlib>
#include <vector>
//#include <stdlib>
#include <cmath>
#include <sstream>


#include "types.h"

#include "tetracut.h"

#include "triheader.h"

#include "triconverter.h"

#define NUM_FLOATS_VERTEX 6
#define NUM_FLOATS_VELOCITY 3

using namespace std;

const int TriConverter::VELX =  0x01;
const int TriConverter::VELY =  0x02;
const int TriConverter::VELZ =  0x04;
const int TriConverter::POS  =  0x08;
const int TriConverter::DENS =  0x10;


TriConverter::TriConverter(int imagesize,
                           float boxsize,
                            int maxNumTriangles
                           ){
    
    
    //printf("%s %s\n", prefix.c_str(), outputbasename.c_str());
    imagesize_ = imagesize;
    boxsize_ = boxsize;
    dz_ = boxsize_ / imagesize_;
    startz_ = 0;
    numplanes_ = imagesize_;
    numTrianglePlanes = new int[imagesize_];
    maxNumTriangles_ = maxNumTriangles;
    currentTriNum_ = 0;
    
    
    trianglePlaneIds_.reserve(maxNumTriangles_ + 100);
    vertexData_.reserve((maxNumTriangles_ + 100) * 6);  //every triangle has 3 vertexs, each vertex has 2 data point
    densityData_.reserve(maxNumTriangles_ + 100);       //each triangle has a single density
    totalTriangles_ = 0;
    
    for(int i = 0; i < imagesize_; i ++){
        //vertexIds_[i] = 0;
        numTrianglePlanes[i] = 0;
    }
}

TriConverter::~TriConverter(){
    delete[] numTrianglePlanes;
}




vector<int> & TriConverter::getTrianglePlaneIds(){
    return trianglePlaneIds_;
}//return a vector of the triangle ids
vector<float> & TriConverter::getVertex(){
    return vertexData_;
}//get a float array of the vertexes
vector<float> & TriConverter::getDensity(){
    return densityData_;
}//get a float vector of densities
int * TriConverter::getNumTrisInPlanes(){
    return numTrianglePlanes;
}//get a array of number of triangles in each plane
bool TriConverter::isReachMax(){
    return (currentTriNum_ >= maxNumTriangles_);
}//whether the numoftris reach maximum
void TriConverter::reset(){
    currentTriNum_ = 0;
    trianglePlaneIds_.clear();
    vertexData_.clear();
    densityData_.clear();
    memset(numTrianglePlanes, 0, sizeof(int) * imagesize_);
    totalTriangles_ = 0;
}                          //clear memories
int TriConverter::getTotalTriangles(){
    return totalTriangles_;
}

void TriConverter::process(Tetrahedron * tetras, int numTetras){
    for(int i = 0; i < numTetras; i++){
        this->process(tetras[i]);
    }
}

void TriConverter::process(Tetrahedron & tetra){
    cutter.setTetrahedron(&tetra);
    
    if(0 > tetra.v4.z || tetra.v1.z > boxsize_){
        return;
    }
    
    int starti = max(floor((tetra.v1.z)/ dz_), 0.0f);
    int endi = min(ceil((tetra.v4.z  - startz_) / dz_),
                   (float)numplanes_);

    for(int i = starti; i < endi; i++){
        float z = startz_ + dz_ * i;

        //test
        int tris = cutter.cut(z);
        for(int j = 0; j < tris; j++){
            
            trianglePlaneIds_.push_back(i);
            vertexData_.push_back(cutter.getTriangle(j).a.x);
            vertexData_.push_back(cutter.getTriangle(j).a.y);
            vertexData_.push_back(cutter.getTriangle(j).b.x);
            vertexData_.push_back(cutter.getTriangle(j).b.y);
            vertexData_.push_back(cutter.getTriangle(j).c.x);
            vertexData_.push_back(cutter.getTriangle(j).c.y);
            float dens = 1.0 / tetra.volume / 6.0;
            if(isnan(dens)){
                dens = 0.0;
            }
            densityData_.push_back(dens);
            numTrianglePlanes[i] ++;
            currentTriNum_ ++;
            totalTriangles_ ++;
        }
    }
}



