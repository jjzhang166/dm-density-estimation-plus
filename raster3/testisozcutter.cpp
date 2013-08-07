#include <cstdio>
#include "tetracut.h"
#include "tetrahedron.h"

using namespace std;
int main(){
    IsoZCutter cutter;
    Tetrahedron tetra;
    tetra.v1.x = 0;
    tetra.v1.y = 0;
    tetra.v1.z = 0;
    tetra.v2.x = 0;
    tetra.v2.y = 1;
    tetra.v2.z = 1;
    tetra.v3.x = 1;
    tetra.v3.y = 0;
    tetra.v3.z = 1;
    tetra.v4.x = 1;
    tetra.v4.y = 1;
    tetra.v4.z = 0;
    
    cutter.setTetrahedron(&tetra);
    
    printf("%f %f %f\n", tetra.v1.x, tetra.v1.y, tetra.v1.z);
    printf("%f %f %f\n", tetra.v2.x, tetra.v2.y, tetra.v2.z);
    printf("%f %f %f\n", tetra.v3.x, tetra.v3.y, tetra.v3.z);
    printf("%f %f %f\n", tetra.v4.x, tetra.v4.y, tetra.v4.z);
    
    int tris;
    
    tris = cutter.cut(0);
    printf("Tris: %d\n", tris);
    for(int i = 0; i < tris; i++){
        printf("Triangle: %d\n", i);
        printf("%f %f %f\n", cutter.getTrangle(i).a.x, cutter.getTrangle(i).a.y, cutter.getTrangle(i).a.z);
        printf("%f %f %f\n", cutter.getTrangle(i).b.x, cutter.getTrangle(i).b.y, cutter.getTrangle(i).b.z);
        printf("%f %f %f\n", cutter.getTrangle(i).c.x, cutter.getTrangle(i).c.y, cutter.getTrangle(i).c.z);
        printf("\n");
    }
    
    tris = cutter.cut(0.3);
    printf("Tris: %d\n", tris);
    for(int i = 0; i < tris; i++){
        printf("Triangle: %d\n", i);
        printf("%f %f %f\n", cutter.getTrangle(i).a.x, cutter.getTrangle(i).a.y, cutter.getTrangle(i).a.z);
        printf("%f %f %f\n", cutter.getTrangle(i).b.x, cutter.getTrangle(i).b.y, cutter.getTrangle(i).b.z);
        printf("%f %f %f\n", cutter.getTrangle(i).c.x, cutter.getTrangle(i).c.y, cutter.getTrangle(i).c.z);
        printf("\n");
    }
    tris = cutter.cut(0.5);
    printf("Tris: %d\n", tris);
    for(int i = 0; i < tris; i++){
        printf("Triangle: %d\n", i);
        printf("%f %f %f\n", cutter.getTrangle(i).a.x, cutter.getTrangle(i).a.y, cutter.getTrangle(i).a.z);
        printf("%f %f %f\n", cutter.getTrangle(i).b.x, cutter.getTrangle(i).b.y, cutter.getTrangle(i).b.z);
        printf("%f %f %f\n", cutter.getTrangle(i).c.x, cutter.getTrangle(i).c.y, cutter.getTrangle(i).c.z);
        printf("\n");
    }
    
    tris = cutter.cut(0.7);
    printf("Tris: %d\n", tris);
    for(int i = 0; i < tris; i++){
        printf("Triangle: %d\n", i);
        printf("%f %f %f\n", cutter.getTrangle(i).a.x, cutter.getTrangle(i).a.y, cutter.getTrangle(i).a.z);
        printf("%f %f %f\n", cutter.getTrangle(i).b.x, cutter.getTrangle(i).b.y, cutter.getTrangle(i).b.z);
        printf("%f %f %f\n", cutter.getTrangle(i).c.x, cutter.getTrangle(i).c.y, cutter.getTrangle(i).c.z);
        printf("\n");
    }
    
    tris = cutter.cut(1);
    printf("Tris: %d\n", tris);
    for(int i = 0; i < tris; i++){
        printf("Triangle: %d\n", i);
        printf("%f %f %f\n", cutter.getTrangle(i).a.x, cutter.getTrangle(i).a.y, cutter.getTrangle(i).a.z);
        printf("%f %f %f\n", cutter.getTrangle(i).b.x, cutter.getTrangle(i).b.y, cutter.getTrangle(i).b.z);
        printf("%f %f %f\n", cutter.getTrangle(i).c.x, cutter.getTrangle(i).c.y, cutter.getTrangle(i).c.z);
        printf("\n");
    }
}