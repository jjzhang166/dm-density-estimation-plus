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
                           string outputbasename,
                           int outputBufferSize
                           ){
    
    isVelX_ = false;
    isVelY_ = false;
    isVelZ_ = false;
    isPosition_ = false;
    isDensity_ = false;
    velXbuffer_ = NULL;
    velYbuffer_ = NULL;
    velZbuffer_ = NULL;
    vertexbuffer_ = NULL;
    densbuffer_ = NULL;
    
    //printf("%s %s\n", prefix.c_str(), outputbasename.c_str());
    imagesize_ = imagesize;
    boxsize_ = boxsize;
    dz_ = boxsize_ / imagesize_;
    startz_ = 0;
    numplanes_ = imagesize_;

    outputBufferSize_ = outputBufferSize;
    prefix_ = "";
    outputBaseName_ = outputbasename;
    

    vertexIds_ = new int[imagesize_];
    totalTriangles_ = new int[imagesize_];
    
    for(int i = 0; i < imagesize_; i ++){
        vertexIds_[i] = 0;
        totalTriangles_[i] = 0;
    }
}

TriConverter::~TriConverter(){
    delete vertexIds_;
    
    if(isPosition_){
        if(vertexbuffer_ != NULL){
            delete vertexbuffer_;
        }
    }
    
    if(isDensity_){
        if(densbuffer_ != NULL){
            delete densbuffer_;
        }
    }
    
    if(isVelX_){
        if(velXbuffer_ != NULL){
            delete velXbuffer_;
        }
    }
    if(isVelY_){
        if(velYbuffer_ != NULL){
            delete velYbuffer_;
        }
    }
    if(isVelZ_){
        if(velZbuffer_ != NULL){
            delete velZbuffer_;
        }
    }
}


void TriConverter::writeToFile(int type,
                               int i,
                               ios_base::openmode mode,
                               const char* s,
                               streamsize n,
                               bool isHeader
                               ){
    stringstream ss;
    ss << i;
    string filename = "";

    if(isPosition_ && type == POS){
        filename = prefix_ + outputBaseName_ + "."TRIFILESUFFIX"." + ss.str();
    }else if(isDensity_ && type == DENS){
        filename = prefix_ + outputBaseName_ + "."DENFILESUFFIX"." + ss.str();
    }else if(isVelX_ && type == VELX){
        filename = prefix_ + outputBaseName_ + "."VELXFILESUFFIX"." + ss.str();
    }else if(isVelY_ && type == VELY){
        filename = prefix_ + outputBaseName_ + "."VELYFILESUFFIX"." + ss.str();
    }else if(isVelZ_ && type == VELZ){
        filename = prefix_ + outputBaseName_ + "."VELZFILESUFFIX"." + ss.str();
    }else{
        return;
    }
    
    fstream outDataStream;
    outDataStream.open(filename.c_str(), mode);
    
    if(isHeader){
        outDataStream.seekp(0, ios::beg);
    }
    
    if(!outDataStream.good()){
        printf("Bad file: %s!\n", filename.c_str());
        exit(1);
    }
    outDataStream.write(s, n);
    outDataStream.close();

}

//render the i-th buffer
void TriConverter::outputPlane(int i){
    
    writeToFile(POS,
                i,
                ios::out | ios::binary | ios::app,
                (char * )(vertexbuffer_ + NUM_FLOATS_VERTEX * outputBufferSize_ * i),
                sizeof(float) * vertexIds_[i] * NUM_FLOATS_VERTEX
                );
    writeToFile(DENS,
                i,
                ios::out | ios::binary | ios::app,
                (char * )(densbuffer_ +  outputBufferSize_ * i),
                sizeof(float) * vertexIds_[i]
                );
  
    writeToFile(VELX,
                i,
                ios::out | ios::binary | ios::app,
                (char * )(velXbuffer_ +  NUM_FLOATS_VELOCITY * outputBufferSize_ * i),
                sizeof(float) * vertexIds_[i] * NUM_FLOATS_VELOCITY
                );
    
    writeToFile(VELY,
                i,
                ios::out | ios::binary | ios::app,
                (char * )(velYbuffer_ +  NUM_FLOATS_VELOCITY * outputBufferSize_ * i),
                sizeof(float) * vertexIds_[i] * NUM_FLOATS_VELOCITY
                );
    
    writeToFile(VELZ,
                i,
                ios::out | ios::binary | ios::app,
                (char * )(velZbuffer_ +  NUM_FLOATS_VELOCITY * outputBufferSize_ * i),
                sizeof(float) * vertexIds_[i] * NUM_FLOATS_VELOCITY
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

        //test
        int tris = cutter.cut(z);
        for(int j = 0; j < tris; j++){
            
            
            if(isPosition_){
                vertexbuffer_[outputBufferSize_ * i * NUM_FLOATS_VERTEX + vertexIds_[i] * NUM_FLOATS_VERTEX + 0]
                = cutter.getTriangle(j).a.x;
                vertexbuffer_[outputBufferSize_ * i * NUM_FLOATS_VERTEX + vertexIds_[i] * NUM_FLOATS_VERTEX + 1]
                = cutter.getTriangle(j).a.y;
                vertexbuffer_[outputBufferSize_ * i * NUM_FLOATS_VERTEX + vertexIds_[i] * NUM_FLOATS_VERTEX + 2]
                = cutter.getTriangle(j).b.x;
                vertexbuffer_[outputBufferSize_ * i * NUM_FLOATS_VERTEX + vertexIds_[i] * NUM_FLOATS_VERTEX + 3]
                = cutter.getTriangle(j).b.y;
                vertexbuffer_[outputBufferSize_ * i * NUM_FLOATS_VERTEX + vertexIds_[i] * NUM_FLOATS_VERTEX + 4]
                = cutter.getTriangle(j).c.x;
                vertexbuffer_[outputBufferSize_ * i * NUM_FLOATS_VERTEX + vertexIds_[i] * NUM_FLOATS_VERTEX + 5]
                = cutter.getTriangle(j).c.y;
            }

            if(isDensity_){
                densbuffer_[vertexIds_[i] + outputBufferSize_ * i] = 1.0 / tetra.volume;
            }
            
            if(isVelX_){
                velXbuffer_[outputBufferSize_ * i * NUM_FLOATS_VELOCITY + vertexIds_[i] * NUM_FLOATS_VELOCITY + 0] = cutter.getTriangle(j).val1.x / tetra.volume;
                
                velXbuffer_[outputBufferSize_ * i * NUM_FLOATS_VELOCITY + vertexIds_[i] * NUM_FLOATS_VELOCITY + 1] = cutter.getTriangle(j).val2.x / tetra.volume;
                
                velXbuffer_[outputBufferSize_ * i * NUM_FLOATS_VELOCITY + vertexIds_[i] * NUM_FLOATS_VELOCITY + 2] = cutter.getTriangle(j).val3.x / tetra.volume;
            }
            
            if(isVelY_){
                velYbuffer_[outputBufferSize_ * i * NUM_FLOATS_VELOCITY + vertexIds_[i] * NUM_FLOATS_VELOCITY + 0] = cutter.getTriangle(j).val1.y / tetra.volume;
                
                velYbuffer_[outputBufferSize_ * i * NUM_FLOATS_VELOCITY + vertexIds_[i] * NUM_FLOATS_VELOCITY + 1] = cutter.getTriangle(j).val2.y / tetra.volume;
                
                velYbuffer_[outputBufferSize_ * i * NUM_FLOATS_VELOCITY + vertexIds_[i] * NUM_FLOATS_VELOCITY + 2] = cutter.getTriangle(j).val3.y / tetra.volume;
            }
            
            if(isVelZ_){
                velZbuffer_[outputBufferSize_ * i * NUM_FLOATS_VELOCITY + vertexIds_[i] * NUM_FLOATS_VELOCITY + 0] = cutter.getTriangle(j).val1.z / tetra.volume;
                
                velZbuffer_[outputBufferSize_ * i * NUM_FLOATS_VELOCITY + vertexIds_[i] * NUM_FLOATS_VELOCITY + 1] = cutter.getTriangle(j).val2.z / tetra.volume;
                
                velZbuffer_[outputBufferSize_ * i * NUM_FLOATS_VELOCITY + vertexIds_[i] * NUM_FLOATS_VELOCITY + 2] = cutter.getTriangle(j).val3.z / tetra.volume;
            }
            
            vertexIds_[i] ++;
            totalTriangles_[i] ++;
            
            if(vertexIds_[i] >= outputBufferSize_){
                outputPlane(i);
            }
        }
    }
}

void TriConverter::setOutput(int outCode){
    bool res[] = {false, true};
    isVelX_ =       res[ outCode % 2];
    isVelY_ =       res[(outCode / 2) % 2];
    isVelZ_ =       res[(outCode / 4) % 2];
    isPosition_ =   res[(outCode / 8) % 2];
    isDensity_ =    res[(outCode / 16)% 2];

    
    
    if(isVelX_){
        if(velXbuffer_ != NULL){
            delete velXbuffer_;
        }
        velXbuffer_ = new float[NUM_FLOATS_VELOCITY *
                                outputBufferSize_ *
                                numplanes_];
    }
    if(isVelY_){
        if(velYbuffer_ != NULL){
            delete velYbuffer_;
        }
        velYbuffer_ = new float[NUM_FLOATS_VELOCITY *
                                outputBufferSize_ *
                                numplanes_];
    }
    if(isVelZ_){
        if(velZbuffer_ != NULL){
            delete velZbuffer_;
        }
        velZbuffer_ = new float[NUM_FLOATS_VELOCITY *
                                outputBufferSize_ *
                                numplanes_];
    }
    
    if(isPosition_){
        if(vertexbuffer_ != NULL){
            delete vertexbuffer_;
        }
            vertexbuffer_   = new float[NUM_FLOATS_VERTEX *
                                        outputBufferSize_ *
                                        numplanes_];
    }
    
    if(isDensity_){
        if(densbuffer_ != NULL){
            delete densbuffer_;
        }
        densbuffer_ = new float[1 * outputBufferSize_ *
                                    numplanes_];
    }
    
    
    for(int i = 0; i < imagesize_; i++){
        TriHeader header;
        header.NumTriangles = 0;
        header.boxSize = boxsize_;
        header.z_id = i;
        header.z_coor = (double) i / (double) imagesize_ * boxsize_;;
        header.fileID = i;
        header.NumFiles = imagesize_;
        
        if(isPosition_)
        writeToFile(POS,
                    i,
                    ios::out | ios::binary | ios::app,
                    (char * )((char *) &header),
                    sizeof(header)
                    );
        
        
        if(isDensity_)
        writeToFile(DENS,
                    i,
                    ios::out | ios::binary | ios::app,
                    (char * )((char *) &header),
                    sizeof(header)
                    );
        
        
        if(isVelX_)
        writeToFile(VELX,
                    i,
                    ios::out | ios::binary | ios::app,
                    (char * )((char *) &header),
                    sizeof(header)
                    );
        
        
        if(isVelY_)
        writeToFile(VELY,
                    i,
                    ios::out | ios::binary | ios::app,
                    (char * )((char *) &header),
                    sizeof(header)
                    );
        
        
        if(isVelZ_)
        writeToFile(VELZ,
                    i,
                    ios::out | ios::binary | ios::app,
                    (char * )((char *) &header),
                    sizeof(header)
                    );
    }
    
}

void TriConverter::finish(){
    
    for(int i = 0; i < numplanes_; i++){
        //printf("%d \n", totalTriangles_[i]);
        if(vertexIds_[i] > 0){
            outputPlane(i);
        }
        
        TriHeader header;
        header.NumTriangles = totalTriangles_[i];
        header.boxSize = boxsize_;

        header.boxSize = boxsize_;
        header.z_id = i;
        header.z_coor = (double) i / (double) imagesize_ * boxsize_;;
        header.fileID = i;
        header.NumFiles = imagesize_;
        
        if(isPosition_)
        writeToFile(POS,
                    i,
                    ios::out | ios::binary | ios::in,
                    (char * )((char *) &header),
                    sizeof(header),
                    true
                    );
        
        if(isDensity_)
        writeToFile(DENS,
                    i,
                    ios::out | ios::binary | ios::in,
                    (char * )((char *) &header),
                    sizeof(header),
                    true
                    );
        
        if(isVelX_)
        writeToFile(VELX,
                    i,
                    ios::out | ios::binary | ios::in,
                    (char * )((char *) &header),
                    sizeof(header),
                    true
                    );
        
        if(isVelY_)
        writeToFile(VELY,
                    i,
                    ios::out | ios::binary | ios::in,
                    (char * )((char *) &header),
                    sizeof(header),
                    true
                    );
        
        if(isVelZ_)
        writeToFile(VELZ,
                    i,
                    ios::out | ios::binary | ios::in,
                    (char * )((char *) &header),
                    sizeof(header),
                    true
                    );
        
        
        
    }
    
}

