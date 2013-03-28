#include <cmath>
#include "accretion.h"
#include "triangle.h"
#include "tetracut.h"

using namespace std;

double accretion_sphere_rate(int counts, Point * posdata, Point * veldata, double mass, Point &halocenter, double r1, double r2){
    double volum = 4.0 * PI / 3.0 * (r2 * r2 * r2 - r1 * r1 * r1);
    double accr = 0.0;
    for(int i = 0; i < counts; i ++){
        Point rvec = halocenter - posdata[i];
        double r = sqrt(rvec.dot(rvec));
        rvec = rvec / r;
        if(r > r1 && r < r2){
            accr += rvec.dot(veldata[i]);
        }
    }
    return accr * mass / volum * 4.0 * PI * r1 * r1;
}

double getRadius(const Point &v1, const Point &v2){
    Point rvec = v1-v2;
    return sqrt(rvec.dot(rvec));
}

double accretion_tetra_rate(
                    TetraStreamer &tetrastreamer, 
                    double mass, 
                    Point &halocenter, 
                    double r){
    double accr = 0.0;
    tetrastreamer.reset();
    IsoCutter cutter;
    while(tetrastreamer.hasNext()){
        Tetrahedron * tetras;
        int nums;
        tetras = tetrastreamer.getNext(nums);
        for(int i = 0; i < nums; i++){
            cutter.setTetrahedron(&tetras[i]);
            cutter.setValues(getRadius(tetras[i].v1, halocenter),
                             getRadius(tetras[i].v2, halocenter),
                             getRadius(tetras[i].v3, halocenter),
                             getRadius(tetras[i].v4, halocenter)
                            );
            int nu_tri = cutter.cut(r);
            
            /*if(nu_tri > 0){
                printf("Radius: %d %f %f %f %f %f\n", nu_tri, r,
                            getRadius(tetras[i].v1, halocenter),
                            getRadius(tetras[i].v2, halocenter),
                            getRadius(tetras[i].v3, halocenter),
                            getRadius(tetras[i].v4, halocenter));
            }*/


            for(int j = 0; j < nu_tri; j++){
                Triangle3d t = cutter.getTrangle(j);
                double area = t.getArea();
                //printf("\n");
                /*printf("%f %f %f\n", t.a.x, t.a.y, t.a.z);
                printf("%f %f %f\n", t.b.x, t.b.y, t.b.z);
                printf("%f %f %f\n", t.c.x, t.c.y, t.c.z);*/

                accr += area;
                //printf("Area: %f %f\n", area, area/(4*PI*r*r));
                //printf("\n");

            }
        }
    }
    return accr;
}

