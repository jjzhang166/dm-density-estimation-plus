#include <cstdlib>
#include <cstring>
#include <cmath>
#include "cic.h"

using namespace std;

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
    int indxmin = (int) floor((pos[0] - dx / 2.0) / dx);
    int indymin = (int) floor((pos[1] - dx / 2.0) / dx);
    int indzmin = (int) floor((pos[2] - dx / 2.0) / dx);
    
    int indxmax = (int) ceil((pos[0] + dx / 2.0) / dx);
    int indymax = (int) ceil((pos[1] + dx / 2.0) / dx);
    int indzmax = (int) ceil((pos[2] + dx / 2.0) / dx);
    
    for(int i = indxmin; i < indxmax; i++){
        for(int j = indymin; j < indymax; j++){
            for(int k = indzmin; k < indzmax; k++){
                int ind = i + j * gridsize_ + k * gridsize_ * gridsize_;
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