#include <unistd.h>
#include <sys/time.h>

#include "tetracut.h"
#include "isoplane.h"

/*bool iso_cut_line(REAL _isoval, Point &v1, Point &v2, Point2d &trianglev){
    
    //--- Check if the isodensity lies between the two vertices
    //printf("%f %f %f\n", v1.z, v2.z, _isoval);
    if (((v1.z - _isoval) * (v2.z - _isoval) <= 0) && (v1.z != v2.z)) {
        //--- Parameter
        REAL t_par;
        //--- Get two point parametric curve of the line along the two vertices
        t_par = (_isoval - v1.z) / (v2.z - v1.z);
        
        trianglev.x = (1-t_par) * v1.x  + t_par * v2.x;
        trianglev.y = (1-t_par) * v1.y  + t_par * v2.y;
        //trianglev.z = (1-t_par) * v1.z  + t_par * v2.z;
        return true;
        
    }
    return false;
    
}


bool testTriangle(Point2d &a, Point2d &b, Point2d &c, Point2d &d){
    if(((c.x -a.x)*(b.y-a.y) - (c.y-a.y)*(b.x-a.x)) *
       ((d.x -a.x)*(b.y-a.y) - (d.y-a.y)*(b.x-a.x)) < 0)
        return true;
    else
        return false;
}
*/

void getTriangles(REAL isoval, int & count,
                  Triangle * triangles,
                  Tetrahedron &tetra){
    Point zero;
    zero.x = 0;
    zero.y = 0;
    zero.z = 0;
    
    //return;
    
    static IsoCutter cutter;
    cutter.setTetrahedron(&tetra);
    cutter.setValues(tetra.v1.z, tetra.v2.z, tetra.v3.z, tetra.v4.z);
    int num = cutter.cut(isoval);
    
    //return;
    
    //rettetra;
    
    for(int i = 0; i < num; i++){
        Triangle3d& rettetra = cutter.getTrangle(i);
        triangles[count].a = rettetra.a;
        triangles[count].b = rettetra.b;
        triangles[count].c = rettetra.c;
        triangles[count].val1 = zero;
        triangles[count].val2 = zero;
        triangles[count].val3 = zero;
        triangles[count].val1.x = tetra.invVolume;
        triangles[count].val2.x = tetra.invVolume;
        triangles[count].val3.x = tetra.invVolume;
        count ++;

    }
}

void TetraIsoPlane::convertTetras2IsoPlane(){
    
    Triangle * triangles = isoplane_;
    IndTetrahedronManager& tetramanager = tetraStream_->getCurrentIndTetraManager();;

    int count  = 0;
    timeval timediff;
    double t1, t0 = 0;

    //return;
    while((count < isoplane_mem_size_ - 1) && (current_tetra_num_ < total_tetra_num_)){

        //test
        /*if(tetramanager.posa(tetras[current_tetra_num_]).x >= 0 ||
           tetramanager.posb(tetras[current_tetra_num_]).x >= 0 ||
           tetramanager.posc(tetras[current_tetra_num_]).x >= 0 ||
           tetramanager.posd(tetras[current_tetra_num_]).x >= 0){
            printf("%d %d %d %d\n", tetras[current_tetra_num_].ind1, tetras[current_tetra_num_].ind2, tetras[current_tetra_num_].ind3, tetras[current_tetra_num_].ind4);
            printf("%f %f %f\n", tetramanager.posa(tetras[current_tetra_num_]).x, tetramanager.posa(tetras[current_tetra_num_]).y, tetramanager.posa(tetras[current_tetra_num_]).z);
            printf("%f %f %f\n", tetramanager.posb(tetras[current_tetra_num_]).x, tetramanager.posb(tetras[current_tetra_num_]).y, tetramanager.posb(tetras[current_tetra_num_]).z);
            printf("%f %f %f\n", tetramanager.posc(tetras[current_tetra_num_]).x, tetramanager.posc(tetras[current_tetra_num_]).y, tetramanager.posc(tetras[current_tetra_num_]).z);
            printf("%f %f %f\n", tetramanager.posd(tetras[current_tetra_num_]).x, tetramanager.posd(tetras[current_tetra_num_]).y, tetramanager.posd(tetras[current_tetra_num_]).z);
        }*/
        
        //ignore the tetrahedrons that is not usable
        if(tetramanager.posa(tetras[current_tetra_num_]).x < 0 ||
           tetramanager.posb(tetras[current_tetra_num_]).x < 0 ||
           tetramanager.posc(tetras[current_tetra_num_]).x < 0 ||
           tetramanager.posd(tetras[current_tetra_num_]).x < 0){
            //This tetrahedron is ignored
        }else{
        
            int temp_num_tetra = tetramanager.getNumPeriodical(tetras[current_tetra_num_]);
            Tetrahedron * period_tetras = tetramanager.getPeroidTetras(tetras[current_tetra_num_]);
            for(int j = 0; j<temp_num_tetra; j++){
            
                gettimeofday(&timediff, NULL);
                t0 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
            
                getTriangles(isovalue_, count, triangles, period_tetras[j]);
            
                gettimeofday(&timediff, NULL);
                t1 = timediff.tv_sec + timediff.tv_usec / 1.0e6;
                cuttingtime_ += t1 - t0;
            
                if(count >= isoplane_mem_size_ - 1){
                    break;
                }
            }
        }
        current_tetra_num_ ++;
        //printf("%d %d\n", count, current_tetra_num_);
    }
    
    currentIsoPlane_Size_ = count;
}
