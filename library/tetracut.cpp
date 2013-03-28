#include <cstdio>
#include "tetracut.h"

using namespace std;

IsoCutter::IsoCutter(){
    num_tris_ = 0;
}

void IsoCutter::setTetrahedron(Tetrahedron *tetra){
    tetra_ = tetra;
}

void IsoCutter::setValues(
                const REAL v1, 
                const REAL v2, 
                const REAL v3, 
                const REAL v4){
    v1_ = v1;
    v2_ = v2;
    v3_ = v3;
    v4_ = v4;
}

bool IsoCutter::iso_cut_line(REAL _isoval, Point &p1, Point &p2, REAL v1, REAL v2, Point &retp){
    //--- Check if the isodensity lies between the two value
    //printf("%f %f %f\n", v1, v2, _isoval);
    if (((v1 - _isoval) * (v2 - _isoval) <= 0) && ((v1 > v2) || (v2 > v1))) {
        //--- Parameter
        REAL t_par;
        //--- Get two point parametric curve of the line along the two vertices
        t_par = (_isoval - v1) / (v2 - v1);
        
        retp.x = (1-t_par) * p1.x  + t_par * p2.x;
        retp.y = (1-t_par) * p1.y  + t_par * p2.y;
        retp.z = (1-t_par) * p1.z  + t_par * p2.z;
        return true;
        
    }
    return false;

}


//test whether the 3 points forms 2 triangles without intersection: abc, abd 
//a, b, c, d must be in the same plane. Otherwise, the result won't correct
bool IsoCutter::testTriangle(const Point &a, const Point &b, const Point &c, const Point &d){
    Point s1 = (d-a).cross(b-a);
    Point s2 = (c-a).cross(b-a);
    if(s1.dot(s2)< 0){
        /*printf("a cross b\n");
        printf("%f %f %f\n", (a).x, (a).y, (a).z);
        printf("%f %f %f\n", (b).x, (b).y, (b).z);
        printf("%f %f %f\n", (c).x, (c).y, (c).z);
        printf("%f %f %f\n", (d).x, (d).y, (d).z);
        printf("%f %f %f\n", (b-a).x, (b-a).y, (b-a).z);
        printf("%f %f %f\n", (c-a).x, (c-a).y, (c-a).z);
        printf("%f %f %f\n", (d-a).x, (d-a).y, (d-a).z);
        printf("%f %f %f\n", (s1).x, (s1).y, (s1).z);
        printf("%f %f %f\n", (s2).x, (s2).y, (s2).z);*/

        return true;
    }
    else
        return false;
}


int IsoCutter::cut(REAL isoval){
    if((isoval < v1_ && isoval < v2_ && isoval < v3_ && isoval < v4_) ||
       (isoval > v1_ && isoval > v2_ && isoval > v3_ && isoval > v4_)){
        return 0;
    }

    Point verts[6];
    Point zero;
    zero.x = 0;
    zero.y = 0;
    zero.z = 0;
    int vertCount = 0;
    
    //return 0;
    
    if(iso_cut_line(isoval, tetra_->v1, tetra_->v2, v1_, v2_, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra_->v1, tetra_->v3, v1_, v3_, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra_->v1, tetra_->v4, v1_, v4_, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra_->v2, tetra_->v3, v2_, v3_, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra_->v2, tetra_->v4, v2_, v4_, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra_->v3, tetra_->v4, v3_, v4_, verts[vertCount])){
        vertCount ++;
    }
    
    num_tris_ = 0;
    //return num_tris_;
    
    if(vertCount == 3){
        t1_.a = verts[0];
        t1_.b = verts[1];
        t1_.c = verts[2];
        num_tris_ ++;
        return num_tris_;
    }else if(vertCount >= 4){
        int a=0, b=1, c=2, d=3;
        if(!testTriangle(verts[a], verts[b], verts[c], verts[d])){
            a=0; b=2; c=1; d=3;
            if(!testTriangle(verts[a], verts[b], verts[c], verts[d])){
                a=0; b=3; c=2; d=1;
                if(!testTriangle(verts[a], verts[b], verts[c], verts[d])){
                    a=1; b=2; c=0; d=3;
                    if(!testTriangle(verts[a], verts[b], verts[c], verts[d])){
                        a=1; b=3; c=0; d=2;
                        if(!testTriangle(verts[a], verts[b], verts[c], verts[d])){
                            a=2; b=3; c=1; d=0;
                        }
                    }
                }
            }
        }
        
            
        t1_.a = verts[a];
        t1_.b = verts[b];
        t1_.c = verts[c];
        num_tris_ ++;
        t2_.a = verts[a];
        t2_.b = verts[b];
        t2_.c = verts[d];
        num_tris_ ++;
        return num_tris_;
    }
    return num_tris_;
}

Triangle3d& IsoCutter::getTrangle(int i){
    if(i == 0){
        return t1_;
    }else if(i == 1){
        return t2_;
    }else{
        return t1_;
    }
}

