#include <cstdio>
#include <cstdlib>
#include <cstring>
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



TriConverter::TriConverter(int imagesize,
                           float boxsize,
                            int maxNumTriangles,
                           bool isVelocity
                           ){
    
    imagesize_ = imagesize;
    boxsize_ = boxsize;
    dz_ = boxsize_ / imagesize_;
    startz_ = 0;
    numplanes_ = imagesize_;
    numTrianglePlanes = new int[imagesize_];
    maxNumTriangles_ = maxNumTriangles;
    currentTriNum_ = 0;
    
    isVelocity_ = isVelocity;
    //printf("IsVelocity Cao %d\n", isVelocity);
    
    trianglePlaneIds_.reserve(maxNumTriangles_ + 100);
    vertexData_.reserve((maxNumTriangles_ + 100) * 6);  //every triangle has 3 vertexs, each vertex has 2 data point
    densityData_.reserve(maxNumTriangles_ + 100);       //each triangle has a single density
    
    if(isVelocity_){
        velXData_.reserve((maxNumTriangles_ + 100) * 3);
        velYData_.reserve((maxNumTriangles_ + 100) * 3);
        velZData_.reserve((maxNumTriangles_ + 100) * 3);
    }
    
    
    totalTriangles_ = 0;
    
    for(int i = 0; i < imagesize_; i ++){
        //vertexIds_[i] = 0;
        numTrianglePlanes[i] = 0;
    }
}

TriConverter::~TriConverter(){
    delete[] numTrianglePlanes;
}



//return a vector of the triangle ids
vector<int> & TriConverter::getTrianglePlaneIds(){
    return trianglePlaneIds_;
}

//get a float array of the vertexes
vector<float> & TriConverter::getVertex(){
    return vertexData_;
}

//get a float vector of densities
vector<float> & TriConverter::getDensity(){
    return densityData_;
}


vector<float> & TriConverter::getVelocityX(){
    return velXData_;
}

vector<float> & TriConverter::getVelocityY(){
    return velYData_;
}

vector<float> & TriConverter::getVelocityZ(){
    return velZData_;
}

//get a array of number of triangles in each plane
int * TriConverter::getNumTrisInPlanes(){
    return numTrianglePlanes;
}

//whether the numoftris reach maximum
bool TriConverter::isReachMax(){
    return (currentTriNum_ >= maxNumTriangles_);
}



//clear memories
void TriConverter::reset(){
    currentTriNum_ = 0;
    trianglePlaneIds_.clear();
    vertexData_.clear();
    densityData_.clear();
    
    if(isVelocity_){
        velXData_.clear();
        velYData_.clear();
        velZData_.clear();
    }
    
    memset(numTrianglePlanes, 0, sizeof(int) * imagesize_);
    totalTriangles_ = 0;
}


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
            //printf("st000\n");
            if(isVelocity_){
                //printf("st\n");
                velXData_.push_back(cutter.getTriangle(j).val1.x * dens);
                velXData_.push_back(cutter.getTriangle(j).val2.x * dens);
                velXData_.push_back(cutter.getTriangle(j).val3.x * dens);
                
                //test
                velYData_.push_back(cutter.getTriangle(j).val1.y * dens);
                velYData_.push_back(cutter.getTriangle(j).val2.y * dens);
                velYData_.push_back(cutter.getTriangle(j).val3.y * dens);
                
                velZData_.push_back(cutter.getTriangle(j).val1.z * dens);
                velZData_.push_back(cutter.getTriangle(j).val2.z * dens);
                velZData_.push_back(cutter.getTriangle(j).val3.z * dens);
            }
            
            numTrianglePlanes[i] ++;
            currentTriNum_ ++;
            totalTriangles_ ++;
        }
    }
}



