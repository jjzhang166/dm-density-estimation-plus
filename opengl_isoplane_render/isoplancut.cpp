#include "isoplane.h"

bool iso_cut_line(REAL _isoval, Point &v1, Point &v2, Point2d &trianglev){
    
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

void getTriangles(REAL isoval, int & count,
                  Triangle * triangles,
                  Tetrahedron &tetra){
    
    Point2d verts[6];
    Point zero;
    zero.x = 0;
    zero.y = 0;
    zero.z = 0;
    int vertCount = 0;
    
    if(iso_cut_line(isoval, tetra.v1, tetra.v2, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra.v1, tetra.v3, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra.v1, tetra.v4, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra.v2, tetra.v3, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra.v2, tetra.v4, verts[vertCount])){
        vertCount ++;
    }
    
    if(iso_cut_line(isoval, tetra.v3, tetra.v4, verts[vertCount])){
        vertCount ++;
    }
    
    //if(vertCount == 5) printf("%d\n", vertCount);
    //we only cont 2 triangles
    // vertCount == 3: only one triangle
    // vertCount == 4: two triangles
    if(vertCount == 3){
        triangles[count].a = verts[0];
        triangles[count].b = verts[1];
        triangles[count].c = verts[2];
        triangles[count].val1 = zero;
        triangles[count].val2 = zero;
        triangles[count].val3 = zero;
        triangles[count].val1.x = tetra.invVolume;
        triangles[count].val2.x = tetra.invVolume;
        triangles[count].val3.x = tetra.invVolume;
        count ++;
    }
    if(vertCount >= 4){
        int a, b, c, d;
        for(a = 0; a < 4; a++){
            for(b = a + 1; b < 4; b ++){
                for(c = 0; c < 4; c ++){
                    for(d = c + 1; d < 4; d++){
                        if(a != c && b != c && a!=d && b!=d){
                            if(testTriangle(verts[a], verts[b], verts[c], verts[d])){
                                triangles[count].a = verts[a];
                                triangles[count].b = verts[b];
                                triangles[count].c = verts[c];
                                triangles[count].val1 = zero;
                                triangles[count].val2 = zero;
                                triangles[count].val3 = zero;
                                triangles[count].val1.x = tetra.invVolume;
                                triangles[count].val2.x = tetra.invVolume;
                                triangles[count].val3.x = tetra.invVolume;
                                count ++;
                                triangles[count].a = verts[a];
                                triangles[count].b = verts[b];
                                triangles[count].c = verts[d];
                                triangles[count].val1 = zero;
                                triangles[count].val2 = zero;
                                triangles[count].val3 = zero;
                                triangles[count].val1.x = tetra.invVolume;
                                triangles[count].val2.x = tetra.invVolume;
                                triangles[count].val3.x = tetra.invVolume;
                                count ++;
                                //printf("add 4!\n");
                                return;
                            }
                        }
                    }
                }
            }
        }
    }
}

int TetraIsoPlane::convertTetras2IsoPlane(REAL isovalue,
        Triangle * triangles, Tetrahedron * tetras,
                                          int nums){
    int count  = 0;
    //printf("%d\n", nums);
    for(int i = 0; i < nums; i++){
        getTriangles(isovalue, count, triangles, tetras[i]);
    }
    //printf("%d\n", count);
    return count;
}