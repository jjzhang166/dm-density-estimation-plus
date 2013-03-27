#include "triangle.h"
#include "indtetrastream.h"
#include "isoplane.h"

TetraIsoPlane::TetraIsoPlane(IndTetraStream * tetraStream, int isoplane_mem_size){
    tetraStream_ = tetraStream;
    int gridsize = tetraStream->getBlockSize();
    isoPlane_Size_ = 6 * (gridsize) * (gridsize)
    * (gridsize) * 8 * 2;
    isoplane_ = new Triangle[isoPlane_Size_];
    currentIsoPlane_Size_ = 0;
    total_tetra_num_ = tetraStream_->getBlockNumTetra();;
    current_tetra_num_ = 0;
    tetras = NULL;
    isoplane_mem_size_ = isoplane_mem_size + 1;
}

TetraIsoPlane::~TetraIsoPlane(){
    delete isoplane_;
}

bool TetraIsoPlane::hasNext(){
    if(current_tetra_num_ < total_tetra_num_){
        return true;
    }else{
        return false;
    }
}
Triangle * TetraIsoPlane::getNextIsoPlaneBlock(int & num_triangles){
    convertTetras2IsoPlane();
    num_triangles = currentIsoPlane_Size_;
    return isoplane_;
}

//Triangle * TetraIsoPlane::getIsoPlane(int i){
//    loadIsoplane(i);
//    return isoplane_;
//}

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
    total_tetra_num_ = tetraStream_->getBlockNumTetra();
    current_tetra_num_ = 0;
    //tetramanager_ = &(tetraStream_->getCurrentIndTetraManager());
    tetras = tetraStream_ -> getCurrentBlock();
    //convertTetras2IsoPlane();
}
