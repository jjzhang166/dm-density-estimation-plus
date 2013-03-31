#include <cmath>
#include "tetrastream.h"
#include "tetrahedron.h"

using namespace std;

CUDA_CALLABLE_MEMBER IndTetrahedronManager::IndTetrahedronManager(
                                                                  Point * parray,
                                                                  REAL box,
                                                                  bool isVelocity,
                                                                  Point * varray){
    positionArray = parray;
    velocityArray = varray;
    box_ = box;
    isVelocity_ = isVelocity;
}


CUDA_CALLABLE_MEMBER void IndTetrahedronManager::setIsVelocity(bool isVelocity){
    isVelocity_ = isVelocity;
}
CUDA_CALLABLE_MEMBER void IndTetrahedronManager::setPosArray(Point * parray){
    positionArray = parray;
}
CUDA_CALLABLE_MEMBER void IndTetrahedronManager::setVelArray(Point * varray){
    velocityArray = varray;
}
CUDA_CALLABLE_MEMBER void IndTetrahedronManager::setBoxSize(REAL box){
    box_ = box;
}

CUDA_CALLABLE_MEMBER bool IndTetrahedronManager::isInTetra(const IndTetrahedron &tetra_, const Point &p) const{
    
    Tetrahedron t;
    
    t.v1 = positionArray[tetra_.ind1];
    t.v2 = positionArray[tetra_.ind2];
    t.v3 = positionArray[tetra_.ind3];
    t.v4 = positionArray[tetra_.ind4];
    
    return t.isInTetra(p);
}

CUDA_CALLABLE_MEMBER bool IndTetrahedronManager::isInTetra(const IndTetrahedron &tetra_,
                                                           Point &p, double &d0, double &d1, double &d2, double &d3, double &d4) const{
    Tetrahedron t;
    
    t.v1 = positionArray[tetra_.ind1];
    t.v2 = positionArray[tetra_.ind2];
    t.v3 = positionArray[tetra_.ind3];
    t.v4 = positionArray[tetra_.ind4];
    
    return t.isInTetra(p, d0, d1, d2, d3, d4);
}

CUDA_CALLABLE_MEMBER Point& IndTetrahedronManager::posa(const IndTetrahedron &tetra_) const{
    return positionArray[tetra_.ind1];
}
CUDA_CALLABLE_MEMBER Point& IndTetrahedronManager::posb(const IndTetrahedron &tetra_) const{
    return positionArray[tetra_.ind2];
}
CUDA_CALLABLE_MEMBER Point& IndTetrahedronManager::posc(const IndTetrahedron &tetra_) const{
    return positionArray[tetra_.ind3];
}
CUDA_CALLABLE_MEMBER Point& IndTetrahedronManager::posd(const IndTetrahedron &tetra_) const{
    return positionArray[tetra_.ind4];
}


CUDA_CALLABLE_MEMBER Point& IndTetrahedronManager::vela(const IndTetrahedron &tetra_) const{
    return velocityArray[tetra_.ind1];
}
CUDA_CALLABLE_MEMBER Point& IndTetrahedronManager::velb(const IndTetrahedron &tetra_) const{
    return velocityArray[tetra_.ind2];
}
CUDA_CALLABLE_MEMBER Point& IndTetrahedronManager::velc(const IndTetrahedron &tetra_) const{
    return velocityArray[tetra_.ind3];
}
CUDA_CALLABLE_MEMBER Point& IndTetrahedronManager::veld(const IndTetrahedron &tetra_) const{
    return velocityArray[tetra_.ind4];
}

CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::minx(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return min(min(min(v1.x, v2.x), v3.x), v4.x);
}
CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::miny(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return min(min(min(v1.y, v2.y), v3.y), v4.y);
}
CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::minz(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return min(min(min(v1.z, v2.z), v3.z), v4.z);
}
CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::maxx(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return max(max(max(v1.x, v2.x), v3.x), v4.x);
}
CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::maxy(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return max(max(max(v1.y, v2.y), v3.y), v4.y);
}
CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::maxz(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return max(max(max(v1.z, v2.z), v3.z), v4.z);
}


CUDA_CALLABLE_MEMBER int IndTetrahedronManager::getNumPeriodical(const IndTetrahedron &t){
    Tetrahedron &tetra_ = tetras_p[0];
	tetra_.v1 = positionArray[t.ind1];
	tetra_.v2 = positionArray[t.ind2];
	tetra_.v3 = positionArray[t.ind3];
	tetra_.v4 = positionArray[t.ind4];
    
    if(isVelocity_){
     	tetra_.velocity1 = velocityArray[t.ind1];
	    tetra_.velocity2 = velocityArray[t.ind2];
	    tetra_.velocity3 = velocityArray[t.ind3];
	    tetra_.velocity4 = velocityArray[t.ind4];
    }
    
    //printf("%d %d %d %d %f\n", t.ind1, t.ind2, t.ind3, t.ind4, box_);
    tetra_.computeVolume();
    
    //periodical correction:
    int tetra_num = 1;
    int temp_num = 0;
    if(tetra_.maxx() - tetra_.minx() > box_ / 2.0){
        TetraStream::splitTetraX(tetra_, tetras_p[1], box_);
        tetra_num ++;
    }
    temp_num = 0;
    for(int i = 0; i < tetra_num; i++){
        Tetrahedron &t = tetras_p[i];
        if(t.maxy() - t.miny() > box_ / 2.0){
            TetraStream::splitTetraY(t, tetras_p[tetra_num + temp_num], box_);
            temp_num ++;
        }
    }
    tetra_num += temp_num;
    temp_num = 0;
    for(int i = 0; i < tetra_num; i++){
        Tetrahedron &t = tetras_p[i];
        if(t.maxz() - t.minz() > box_ / 2.0){
            TetraStream::splitTetraZ(t, tetras_p[tetra_num + temp_num], box_);
            temp_num ++;
        }
    }
    tetra_num += temp_num;
    
    return tetra_num;
}


CUDA_CALLABLE_MEMBER Tetrahedron * IndTetrahedronManager::getPeroidTetras(const IndTetrahedron &tetra_){
    return tetras_p;
}

CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::getVolume(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return Tetrahedron::getVolume(v1, v2, v3, v4);
}


