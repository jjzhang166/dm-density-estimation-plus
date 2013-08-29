#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cmath>
#include <sstream>


#include "types.h"

#include "tetracut.h"

#include "triconverter.h"

#define NUM_FLOATS_VERTEX 7
#define NUM_FLOATS_VELOCITY 9
#define VELFILESUFFIX "vel"
#define DENFILESUFFIX "den"
#define TRIFILESUFFIX "tri"

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
    
    vertexbuffer_   = new float[NUM_FLOATS_VERTEX * outputBufferSize_ * numplanes_];
    //densbuffer_     = new float[1 * outputBufferSize_ * numplanes_];
    //velocitybuffer_ = new float[NUM_FLOATS_VELOCITY * outputBufferSize_ * numplanes_];
    
    vertexIds_ = new int[imagesize_];
    totalTriangles_ = new int[imagesize_];
    
    for(int i = 0; i < imagesize_; i ++){
        vertexIds_[i] = 0;
        totalTriangles_[i] = 0;
    }
    
    //outputStreams_ = new fstream[imagesize_];
    
    for(int i = 0; i < imagesize_; i++){

        stringstream ss;
        ss << i;
        string trifile = prefix_ + outputBaseName_ + "."TRIFILESUFFIX"." + ss.str();
        //string denfile = prefix_ + outputBaseName_ + "."DENFILESUFFIX"." + ss.str();
        //string velfile = prefix_ + outputBaseName_ + "."VELFILESUFFIX"." + ss.str();

        TriHeader header;
        header.numOfTriangles = 0;
        header.boxSize = boxsize_;
        
        fstream tristream(trifile.c_str(), ios::out | ios::binary);
        if(!tristream.good()){
            printf("Bad file: %s!\n", trifile.c_str());
            exit(1);
        }
        tristream.write((char *) &header, sizeof(header));
        tristream.close();
        
        /*fstream denstream(denfile.c_str(), ios::out | ios::binary);
        if(!denstream.good()){
            printf("Bad file: %s!\n", denfile.c_str());
            exit(1);
        }
        denstream.write((char *) &header, sizeof(header));
        denstream.close();*/
        
        /*fstream velstream(velfile.c_str(), ios::out | ios::binary);
        if(!velstream.good()){
            printf("Bad file: %s!\n", velfile.c_str());
            exit(1);
        }
        velstream.write((char *) &header, sizeof(header));
        velstream.close();*/
    }
}

TriConverter::~TriConverter(){
    delete vertexIds_;
    delete vertexbuffer_;
    //delete densbuffer_;
    //delete velocitybuffer_;
    //delete outputStreams_;
}

//render the i-th buffer
void TriConverter::outputPlane(int i){
    fstream oFstream;
    stringstream ss;
    ss << i;
    
    string trifile = prefix_ + outputBaseName_ + "."TRIFILESUFFIX"." + ss.str();
    //string denfile = prefix_ + outputBaseName_ + "."DENFILESUFFIX"." + ss.str();
    //string velfile = prefix_ + outputBaseName_ + "."VELFILESUFFIX"." + ss.str();
    
    fstream tristream(trifile.c_str(), ios::out | ios::binary | ios::app);
    if(!tristream.good()){
        printf("Bad file: %s!\n", trifile.c_str());
        exit(1);
    }
    /*fstream denstream(denfile.c_str(), ios::out | ios::binary | ios::app);
    if(!denstream.good()){
        printf("Bad file: %s!\n", denfile.c_str());
        exit(1);
    }*/
    
    /*
    fstream velstream(velfile.c_str(), ios::out | ios::binary | ios::app);
    if(!velstream.good()){
        printf("Bad file: %s!\n", velfile.c_str());
        exit(1);
    }*/
    
    //outputStreams_[i]
    tristream.write((char * )(vertexbuffer_ + NUM_FLOATS_VERTEX * outputBufferSize_ * i),
                    sizeof(float) * vertexIds_[i] * NUM_FLOATS_VERTEX
                    );
    /*denstream.write((char * )(densbuffer_ +  outputBufferSize_ * i),
                    sizeof(float) * vertexIds_[i]
                    );*/
    /*velstream.write((char * )(velocitybuffer_ + NUM_FLOATS_VELOCITY * outputBufferSize_ * i),
                    sizeof(float) * vertexIds_[i] * NUM_FLOATS_VELOCITY
                    );*/
    
    vertexIds_[i] = 0;
    
    
    tristream.close();
    //denstream.close();
    //velstream.close();
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
            
            vertexbuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VERTEX + 0]
                = cutter.getTriangle(j).a.x;
            vertexbuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VERTEX + 1]
                = cutter.getTriangle(j).a.y;
            vertexbuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VERTEX + 2]
                = cutter.getTriangle(j).b.x;
            vertexbuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VERTEX + 3]
                = cutter.getTriangle(j).b.y;
            vertexbuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VERTEX + 4]
                = cutter.getTriangle(j).c.x;
            vertexbuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VERTEX + 5]
                = cutter.getTriangle(j).c.y;
            vertexbuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VERTEX + 7]
                = 1.0 / tetra.volume;
            
            /*velocitybuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VELOCITY + 0]
                = cutter.getTriangle(j).val1.x;
            velocitybuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VELOCITY + 1]
                = cutter.getTriangle(j).val1.y;
            velocitybuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VELOCITY + 2]
                = cutter.getTriangle(j).val1.z;
            velocitybuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VELOCITY + 3]
                = cutter.getTriangle(j).val2.x;
            velocitybuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VELOCITY + 4]
                = cutter.getTriangle(j).val2.y;
            velocitybuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VELOCITY + 5]
                = cutter.getTriangle(j).val2.z;
            velocitybuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VELOCITY + 6]
                = cutter.getTriangle(j).val3.x;
            velocitybuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VELOCITY + 7]
                = cutter.getTriangle(j).val3.y;
            velocitybuffer_[vertexIds_[i] + outputBufferSize_ * i * NUM_FLOATS_VELOCITY + 8]
                = cutter.getTriangle(j).val3.z;*/

            //densbuffer_[vertexIds_[i] + outputBufferSize_ * i] = tetra.volume;
            
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
        
        stringstream ss;
        ss << i;
        
        string trifile = prefix_ + outputBaseName_ + "."TRIFILESUFFIX"." + ss.str();
        //string denfile = prefix_ + outputBaseName_ + "."DENFILESUFFIX"." + ss.str();
        //string velfile = prefix_ + outputBaseName_ + "."VELFILESUFFIX"." + ss.str();
        
        fstream tristream(trifile.c_str(), ios::out | ios::binary | ios::app);
        if(!tristream.good()){
            printf("Bad file: %s!\n", trifile.c_str());
            exit(1);
        }
        /*fstream denstream(denfile.c_str(), ios::out | ios::binary | ios::app);
        if(!denstream.good()){
            printf("Bad file: %s!\n", denfile.c_str());
            exit(1);
        }*/
        /*fstream velstream(velfile.c_str(), ios::out | ios::binary | ios::app);
        if(!velstream.good()){
            printf("Bad file: %s!\n", velfile.c_str());
            exit(1);
        }*/
        
        
        TriHeader header;
        header.numOfTriangles = totalTriangles_[i];
        header.boxSize = boxsize_;
        
        
        tristream.seekg(0, tristream.beg);
        tristream.write((char *) &header, sizeof(header));
        tristream.close();
        
        /*denstream.seekg(0, denstream.beg);
        denstream.write((char *) &header, sizeof(header));
        denstream.close();*/
        
        /*velstream.seekg(0, velstream.beg);
        velstream.write((char *) &header, sizeof(header));
        velstream.close();*/
    }
    
}

