#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cmath>
#include <sstream>


#include "types.h"

#include "tetracut.h"

#include "triconverter.h"

using namespace std;







TriConverter::TriConverter(int imagesize,
                     float boxsize,
                     string prefix,
                     string outputbasename,
                     int outputBufferSize
                    ){
    
    //printf("%s %s\n", prefix.c_str(), outputbasename.c_str());
    imagesize_ = imagesize;
    boxsize_ = boxsize;
    dz_ = boxsize_ / imagesize_;
    startz_ = 0;
    numplanes_ = imagesize_;

    outputBufferSize_ = outputBufferSize;
    prefix_ = prefix;
    outputBaseName_ = outputbasename;
    
    //color and vertex
    vertexbuffer_ = new Triangle[outputBufferSize_ * imagesize_];
    vertexIds_ = new int[imagesize_];
    totalTriangles_ = new int[imagesize_];
    
    for(int i = 0; i < imagesize_; i ++){
        vertexIds_[i] = 0;
        totalTriangles_[i] = 0;
    }
    
    outputStreams_ = new fstream[imagesize_];
    
    for(int i = 0; i < imagesize_; i++){
        stringstream ss;
        ss << i;
        string outfilename = prefix_ + outputBaseName_ + "." + ss.str();
        outputStreams_[i].open(outfilename.c_str(), ios::out | ios::binary);
        if(!outputStreams_[i].good()){
            printf("Bad file: %s!\n", outfilename.c_str());
            exit(1);
        }
        TriHeader header;
        header.numOfTriangles = 0;
        header.boxSize = boxsize_;
        outputStreams_[i].write((char *) &header, sizeof(header));
    }
}

TriConverter::~TriConverter(){
    delete vertexIds_;
    delete vertexbuffer_;
    delete outputStreams_;
}

//render the i-th buffer
void TriConverter::outputPlane(int i){
    outputStreams_[i].write(
                            (char * )(vertexbuffer_ + outputBufferSize_ * i),
                            sizeof(Triangle) * vertexIds_[i]
                            );
    vertexIds_[i] = 0;
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

        int tris = cutter.cut(z);
        for(int j = 0; j < tris; j++){
            vertexbuffer_[vertexIds_[i] + outputBufferSize_ * i]
                = cutter.getTriangle(j);
            
            vertexIds_[i] ++;
            totalTriangles_[i] ++;
            
            if(vertexIds_[i] >= outputBufferSize_){
                outputPlane(i);
            }
        }
    }
}


void TriConverter::finish(){
    
    for(int i = 0; i < numplanes_; i++){
        //printf("%d \n", vertexIds_[i]);
        if(vertexIds_[i] > 0){
            outputPlane(i);
        }
        //write header
        TriHeader header;
        header.numOfTriangles = totalTriangles_[i];
        header.boxSize = boxsize_;
        outputStreams_[i].seekg(0, outputStreams_[i].beg);
        outputStreams_[i].write((char *) &header, sizeof(header));
    }
    
}

