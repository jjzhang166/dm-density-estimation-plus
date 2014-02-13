#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include "cic.h"

using namespace std;

#define clamp(a, amin, amax) ((a)>(amax)?(amax):((a) < (amin)? (amin):(a)))

CIC::CIC(double boxSize, int gridsize, bool isVelocityField){
    boxSize_ = boxSize;
    gridsize_ = gridsize;
    isVelocityField_ = isVelocityField;
    dx_ = boxSize_ / ((double) gridsize_);
    densityField = new double[gridsize * gridsize * gridsize];
    velocityXField = NULL;
    velocityYField = NULL;
    velocityZField = NULL;
    if(isVelocityField){
        velocityXField = new double[gridsize * gridsize * gridsize];
        velocityYField = new double[gridsize * gridsize * gridsize];
        velocityZField = new double[gridsize * gridsize * gridsize];
    }
    
    clearGrid();
}

CIC::~CIC(){
    delete[] densityField;
    
    if(isVelocityField_){
        delete velocityXField;
        delete velocityYField;
        delete velocityZField;
    }
}
void CIC::clearGrid(){
    memset((char *) densityField, 0,
           sizeof(double) * gridsize_ * gridsize_ * gridsize_);
    
    if(isVelocityField_){
        memset((char *) velocityXField, 0,
               sizeof(double) * gridsize_ * gridsize_ * gridsize_);
        
        memset((char *) velocityYField, 0,
               sizeof(double) * gridsize_ * gridsize_ * gridsize_);
        
        memset((char *) velocityZField, 0,
               sizeof(double) * gridsize_ * gridsize_ * gridsize_);
    }
}

void CIC::addToGridCells(double * grids, double * pos, double value){
    double dx = dx_;
    int indxmin = clamp((int) floor((pos[0] - dx / 2.0) / dx), 0, gridsize_);
    int indymin = clamp((int) floor((pos[1] - dx / 2.0) / dx), 0, gridsize_);
    int indzmin = clamp((int) floor((pos[2] - dx / 2.0) / dx), 0, gridsize_);
    
    int indxmax = clamp((int) ceil((pos[0] + dx / 2.0) / dx), 0, gridsize_);
    int indymax = clamp((int) ceil((pos[1] + dx / 2.0) / dx), 0, gridsize_);
    int indzmax = clamp((int) ceil((pos[2] + dx / 2.0) / dx), 0, gridsize_);
    
    for(int i = indxmin; i < indxmax; i++){
        for(int j = indymin; j < indymax; j++){
            for(int k = indzmin; k < indzmax; k++){
                int ind = clamp(i + j * gridsize_ + k * gridsize_ * gridsize_,
                                0, gridsize_*gridsize_*gridsize_ - 1);
                grids[ind] += value;
            }
        }
    }
}

void CIC::render_particle(double * pos, double * vel, double mass){
    double partrho = mass / (dx_ * dx_ * dx_);
    addToGridCells(densityField, pos, partrho);
    
    if(isVelocityField_){
        addToGridCells(velocityXField, pos, partrho * vel[0]);
        addToGridCells(velocityYField, pos, partrho * vel[1]);
        addToGridCells(velocityZField, pos, partrho * vel[2]);
    }
}

void CIC::render_particle(double * pos, double * vel, int numParts,
                          double mass){
    for(int i = 0; i < numParts; i++){
        //printf("%d\n", i);
        render_particle(pos + i * 3, vel + i * 3, mass);
    }
}

double * CIC::getDensityField(){
    return densityField;
}

double * CIC::getVelocityXField(){
    return velocityXField;
}

double * CIC::getVelocityYField(){
    return velocityYField;
}

double * CIC::getVelocityZField(){
    return velocityZField;
}


//test
/*int main(){
    double pos[3000];
    for(int i = 0; i < 1000; i++){
        int x = i % 10;
        int y = i / 10 % 10;
        int z = i / 100 % 10;
        
        pos[i * 3 + 0] = x;
        pos[i * 3 + 1] = y;
        pos[i * 3 + 2] = z;
    }
    
    CIC cic(10, 10);
    
    cic.render_particle(pos, NULL, 1000);
    
    double * dens = cic.getDensityField();
    for(int i = 0; i < 1000; i++){
        printf("%f\n", dens[i]);
    }
    
}*/
