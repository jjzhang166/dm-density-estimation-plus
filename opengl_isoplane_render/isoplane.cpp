#include "triangle.h"
#include "indtetrastream.h"
#include "isoplane.h"

TetraIsoPlane::TetraIsoPlane(IndTetraStream * tetraStream){
    tetraStream_ = tetraStream;
    int gridsize = tetraStream->getBlockSize();
    isoPlane_Size_ = 6 * (gridsize) * (gridsize)
    * (gridsize) * 8 * 2;
    isoplane_ = new Triangle[isoPlane_Size_];
    currentIsoPlane_Size_ = 0;
}

TetraIsoPlane::~TetraIsoPlane(){
    delete isoplane_;
}

Triangle * TetraIsoPlane::getCurrentIsoPlane(){
    return isoplane_;
}

Triangle * TetraIsoPlane::getIsoPlane(int i){
    loadIsoplane(i);
    return isoplane_;
}

int TetraIsoPlane::getTriangleNumbers(){
    return currentIsoPlane_Size_;
}

int TetraIsoPlane::getTotalBlockNum(){
    return tetraStream_->getTotalBlockNum();
}

void TetraIsoPlane::setIsoValue(REAL isovalue){
    isovalue_ = isovalue;
}

void TetraIsoPlane::loadIsoplane(int i){
    tetraStream_->loadBlock(i);
    currentIsoPlane_Size_ = convertTetras2IsoPlane(
                        isovalue_, 
                        isoplane_,
                        tetraStream_->getCurrentBlock(),
                        tetraStream_->getCurrentIndTetraManager(),
                        tetraStream_->getBlockNumTetra());
}
