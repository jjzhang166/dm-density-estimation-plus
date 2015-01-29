/*******************************************************
 * This file defines the related procedroes for dealing
 * with the peoridical conditions of index-tetrahedra
 *
 * Author: Lin F. Yang
 * Date: Feb 2014
 * *****************************************************/


#include <cmath>
#include "tetrahedron.h"

using namespace std;

/* Splite a peorical tetrahedra on the X-direction*/
void splitTetraX(Tetrahedron & tetra, Tetrahedron & tetra1, REAL boxsize){
    static Point * vertexs[4];
    static Point * velocity[4];
	static Point * temp_;
	vertexs[0] = &(tetra.v1);
	vertexs[1] = &(tetra.v2);
	vertexs[2] = &(tetra.v3);
	vertexs[3] = &(tetra.v4);
    
    velocity[0] = &(tetra.velocity1);
	velocity[1] = &(tetra.velocity2);
	velocity[2] = &(tetra.velocity3);
	velocity[3] = &(tetra.velocity4);
    
	for(int i = 0; i < 4; i ++){
		for(int j = i+1; j < 4; j++){
			if((vertexs[i]->x) < (vertexs[j]->x)){
				temp_ = vertexs[j];
				vertexs[j] = vertexs[i];
				vertexs[i] = temp_;
                
 				temp_ = velocity[j];
				velocity[j] = velocity[i];
				velocity[i] = temp_;
			}
		}
	}
	vertexs[3]->x += boxsize;
	if(vertexs[3]->x - vertexs[2]->x > boxsize / 2.0){
		vertexs[2]->x += boxsize;
		if(vertexs[2]->x - vertexs[1]->x > boxsize / 2.0){
			vertexs[1]->x += boxsize;
		}
	}
	tetra.computeVolume();
    
	tetra1 = tetra;
	tetra1.v1.x -= boxsize;
	tetra1.v2.x -= boxsize;
	tetra1.v3.x -= boxsize;
	tetra1.v4.x -= boxsize;
	tetra1.computeMaxMin();
}


/* Splite the tetrahedra on y-direction */
void splitTetraY(Tetrahedron & tetra, Tetrahedron & tetra1, REAL boxsize){
    static Point * vertexs[4];
    static Point * velocity[4];
	static Point * temp_;
	vertexs[0] = &(tetra.v1);
	vertexs[1] = &(tetra.v2);
	vertexs[2] = &(tetra.v3);
	vertexs[3] = &(tetra.v4);
    
    velocity[0] = &(tetra.velocity1);
	velocity[1] = &(tetra.velocity2);
	velocity[2] = &(tetra.velocity3);
	velocity[3] = &(tetra.velocity4);
    
	for(int i = 0; i < 4; i ++){
		for(int j = i+1; j < 4; j++){
			if((vertexs[i]->y) < (vertexs[j]->y)){
				temp_ = vertexs[j];
				vertexs[j] = vertexs[i];
				vertexs[i] = temp_;
                
                temp_ = velocity[j];
				velocity[j] = velocity[i];
				velocity[i] = temp_;
			}
		}
	}
	vertexs[3]->y += boxsize;
	if(vertexs[3]->y - vertexs[2]->y > boxsize / 2.0){
		vertexs[2]->y += boxsize;
		if(vertexs[2]->y - vertexs[1]->y > boxsize / 2.0){
			vertexs[1]->y += boxsize;
		}
	}
	tetra.computeVolume();
    
	tetra1 = tetra;
	tetra1.v1.y -= boxsize;
	tetra1.v2.y -= boxsize;
	tetra1.v3.y -= boxsize;
	tetra1.v4.y -= boxsize;
	tetra1.computeMaxMin();
}

/* Splite the tetrahedra on z-direction */
void splitTetraZ(Tetrahedron & tetra, Tetrahedron & tetra1, REAL boxsize){
    static Point * vertexs[4];
	static Point * temp_;
    static Point * velocity[4];
    
	vertexs[0] = &(tetra.v1);
	vertexs[1] = &(tetra.v2);
	vertexs[2] = &(tetra.v3);
	vertexs[3] = &(tetra.v4);
    
    velocity[0] = &(tetra.velocity1);
	velocity[1] = &(tetra.velocity2);
	velocity[2] = &(tetra.velocity3);
	velocity[3] = &(tetra.velocity4);
    
	for(int i = 0; i < 4; i ++){
		for(int j = i+1; j < 4; j++){
			if((vertexs[i]->z) < (vertexs[j]->z)){
				temp_ = vertexs[j];
				vertexs[j] = vertexs[i];
				vertexs[i] = temp_;
                
                temp_ = velocity[j];
				velocity[j] = velocity[i];
				velocity[i] = temp_;
			}
		}
	}
	vertexs[3]->z += boxsize;
	if(vertexs[3]->z - vertexs[2]->z > boxsize / 2.0){
		vertexs[2]->z += boxsize;
		if(vertexs[2]->z - vertexs[1]->z > boxsize / 2.0){
			vertexs[1]->z += boxsize;
		}
	}
	tetra.computeVolume();
    
	tetra1 = tetra;
	tetra1.v1.z -= boxsize;
	tetra1.v2.z -= boxsize;
	tetra1.v3.z -= boxsize;
	tetra1.v4.z -= boxsize;
	tetra1.computeMaxMin();
}






/* This the constrcutor for the index tetrahedron manager */
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

/* Test whether a point is in a index tetrahedra */
CUDA_CALLABLE_MEMBER bool IndTetrahedronManager::isInTetra(const IndTetrahedron &tetra_, const Point &p) const{
    
    Tetrahedron t;
    
    t.v1 = positionArray[tetra_.ind1];
    t.v2 = positionArray[tetra_.ind2];
    t.v3 = positionArray[tetra_.ind3];
    t.v4 = positionArray[tetra_.ind4];
    
    return t.isInTetra(p);
}


/* Return the homogenous coordinates for index tetrahedra */
CUDA_CALLABLE_MEMBER bool IndTetrahedronManager::isInTetra(const IndTetrahedron &tetra_,
                                                           Point &p, 
                                                           double &d0, 
                                                           double &d1, 
                                                           double &d2, 
                                                           double &d3, 
                                                           double &d4) const{
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


/* Velocity of the four points */
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
    return fmin(fmin(fmin(v1.x, v2.x), v3.x), v4.x);
}

CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::miny(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return fmin(fmin(fmin(v1.y, v2.y), v3.y), v4.y);
}

CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::minz(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return fmin(fmin(fmin(v1.z, v2.z), v3.z), v4.z);
}

CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::maxx(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return fmax(fmax(fmax(v1.x, v2.x), v3.x), v4.x);
}

CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::maxy(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return fmax(fmax(fmax(v1.y, v2.y), v3.y), v4.y);
}

CUDA_CALLABLE_MEMBER REAL IndTetrahedronManager::maxz(const IndTetrahedron &tetra_) const{
    Point &v1 = positionArray[tetra_.ind1];
    Point &v2 = positionArray[tetra_.ind2];
    Point &v3 = positionArray[tetra_.ind3];
    Point &v4 = positionArray[tetra_.ind4];
    return fmax(fmax(fmax(v1.z, v2.z), v3.z), v4.z);
}

/* Compute how many peoridical tetrahedras are there.
 * Calling this function will computes the peoridical tetrahedra at the same time*/
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
    
    tetra_.computeVolume();
    
    //periodical correction:
    int tetra_num = 1;
    
    int temp_num = 0;
    if(tetra_.maxx() - tetra_.minx() > box_ / 2.0){
        splitTetraX(tetra_, tetras_p[1], box_);
        tetra_num ++;
    }

    temp_num = 0;
    for(int i = 0; i < tetra_num; i++){
        Tetrahedron &t = tetras_p[i];
        if(t.maxy() - t.miny() > box_ / 2.0){
            splitTetraY(t, tetras_p[tetra_num + temp_num], box_);
            temp_num ++;
        }
    }

    tetra_num += temp_num;
    temp_num = 0;
    for(int i = 0; i < tetra_num; i++){
        Tetrahedron &t = tetras_p[i];
        if(t.maxz() - t.minz() > box_ / 2.0){
            splitTetraZ(t, tetras_p[tetra_num + temp_num], box_);
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


