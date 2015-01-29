#include <string>
#include <cstdlib>
#include <cmath>
#include <stdint.h>
#include <cstdio>
#include <vector>
#include <fstream>
#include <sstream>
#include "tetrahedron.h"
#include "io_utils.h"
#include "dtetrastream.h"

using namespace std;

void getRedshiftDistoredPoint(Point & target,
                              Point & velocity,
                              Point & distortAxis,
                              double redshift,
                              double boxSize
                              ){
    
    double a = 1.0 / (1.0 + redshift);
    Point displacement = distortAxis
    * velocity.dot(distortAxis)
    * sqrt(a) * RH0;// ; //to kpc/h
    
    target = target + displacement;
    
    target.x = fmod(target.x + boxSize, boxSize);
    target.y = fmod(target.y + boxSize, boxSize);
    target.z = fmod(target.z + boxSize, boxSize);
}



DtetraStream::DtetraStream(std::string basename){
    header_ = readDividerFileHeader(basename, 0);
    
    pos_ = (new Point[header_.numparts ]);
    vel_ = (new Point[header_.numparts]);
    temppos_ = (new Point[header_.numparts]);
    tempvel_ = (new Point[header_.numparts]);
    
    
    inds_ = new int64_t[header_.numparts];
    manager_.setIsVelocity(true);
    manager_.setPosArray((Point *)pos_);
    manager_.setVelArray((Point *)vel_);
    manager_.setBoxSize(header_.boxSize);
    basename_ = basename;
    isReadShiftDistorted_ = false;
}

DtetraStream::~DtetraStream(){
    delete[] temppos_;
    delete[] tempvel_;
    delete[] pos_;
    delete[] vel_;
    delete[] inds_;
}

void DtetraStream::loadBlock(int i){
    if(i < header_.totalfiles){
        header_ = readDividerFileHeader(basename_, i);

        readDividerFilePos(basename_, i, (float *)pos_);
        
        readDividerFileVel(basename_, i, (float *)vel_);
        
        readDividerFileInds(basename_, i, inds_);
        
        //printf("%d \n", header_.numofZgrids);
        
        //sorting
        for(int j = 0; j < header_.numparts; j++){
            int id = inds_[j] - header_.startind;
            
            if(isReadShiftDistorted_){
                getRedshiftDistoredPoint(pos_[j], vel_[j],
                                         distortAxis_,
                                         header_.redshift,
                                         header_.boxSize);
            }
            
            //printf("%d \n", id);
            temppos_[id] = pos_[j];
            //temppos_[id * 3 + 0] = pos_[j * 3 + 0];
            //temppos_[id * 3 + 1] = pos_[j * 3 + 1];
            //temppos_[id * 3 + 2] = pos_[j * 3 + 2];
            
            tempvel_[id] = vel_[j];
            //tempvel_[id * 3 + 0] = vel_[j * 3 + 0];
            //tempvel_[id * 3 + 1] = vel_[j * 3 + 1];
            //tempvel_[id * 3 + 2] = vel_[j * 3 + 2];
        }
        

        
        
        //test
        //for(int j = 0; j < header_.numparts; j++){
        //    printf("Part:%d %f %f %f\n"
        //           "%f %f %f\n", j + header_.startind,
        //           temppos_[j].x, temppos_[j].y, temppos_[j].z,
        //           tempvel_[j].x, tempvel_[j].y, tempvel_[j].z);
        //}
        
        Point * tmp;
        
        tmp = pos_;
        pos_ = temppos_;
        temppos_ = tmp;
        
        tmp = vel_;
        vel_ = tempvel_;
        tempvel_ = tmp;
        
        
        manager_.setPosArray((Point *)pos_);
        manager_.setVelArray((Point *)vel_);
    }
}


IndTetrahedronManager & DtetraStream::getCurrentIndtetraManeger(){
    return manager_;
}

int DtetraStream::getNumTetras(){
    return 6 * header_.gridsize * header_.gridsize * (header_.numofZgrids - 1);
}

divide_header DtetraStream::getHeader(){
    return header_;
}


void DtetraStream::getIndTetra(IndTetrahedron & indtetra, int np){
    static const int ind1i[] = {0, 0, 0, 0, 0, 0};
    static const int ind1j[] = {0, 0, 0, 1, 1, 1};
    static const int ind1k[] = {0, 0, 1, 0, 0, 0};
    
    static const int ind2i[] = {0, 0, 0, 1, 1, 1};
    static const int ind2j[] = {1, 1, 1, 0, 0, 0};
    static const int ind2k[] = {0, 0, 1, 1, 1, 1};
    
    static const int ind3i[] = {0, 1, 1, 1, 1, 1};
    static const int ind3j[] = {0, 0, 0, 1, 1, 1};
    static const int ind3k[] = {1, 1, 1, 1, 1, 0};
    
    static const int ind4i[] = {1, 1, 0, 0, 1, 1};
    static const int ind4j[] = {0, 0, 1, 1, 1, 0};
    static const int ind4k[] = {1, 0, 0, 1, 0, 0};
    
    int vind = np / 6;
    int tind = np % 6;
    

    
    int i = vind % header_.gridsize;
    int j = vind / header_.gridsize % header_.gridsize;
    int k = vind / header_.gridsize / header_.gridsize % header_.gridsize;
    
    //printf("Original: %d %d %d %d\n", vind, i, j, k);
    
    int gridsize = header_.gridsize;
    
    indtetra.ind1 = ((ind1i[tind] + i) % gridsize)
                    + ((ind1j[tind] + j) % gridsize) * gridsize
                    + (ind1k[tind] + k) * gridsize * gridsize;;
    
    indtetra.ind2 =   ((ind2i[tind] + i) % gridsize)
                    + ((ind2j[tind] + j) % gridsize) * gridsize
                    + (ind2k[tind] + k) * gridsize * gridsize;
    
    indtetra.ind3 =   ((ind3i[tind] + i) % gridsize)
                    + ((ind3j[tind] + j) % gridsize) * gridsize
                    + (ind3k[tind] + k) * gridsize * gridsize;
    
    indtetra.ind4 =   ((ind4i[tind] + i) % gridsize)
                    + ((ind4j[tind] + j) % gridsize) * gridsize
                    + (ind4k[tind] + k) * gridsize * gridsize;
}


void DtetraStream::setRedShitDistortion(Point distortAxis){
    isReadShiftDistorted_ = true;
    distortAxis_ = distortAxis;
}
